"""Offline scheduler integration chain (audit phase 5): real SQLite, separate processes.

Every lifecycle step — routing a due prompt, claiming it, completing, failing into
retry backoff, expiring a stuck lease, dead-lettering — runs in its own interpreter
against the same on-disk SQLite database, exactly like a daemon and an agent
taking turns. Clocks are explicit (``now=``) so retry/backoff/lease behavior is
deterministic without sleeping. The user-service registration contract is verified
in-process with the service commands stubbed — a real test never installs a host
service — for both a short and a long command line.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from agent_core import scheduler_service
from agent_core.scheduler import SchedulerStore

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
OWNER = ("session-itg", "agent-itg")

# A frozen base clock: every scenario moves time explicitly from here.
T0 = 1_700_000_000.0

_DRIVER = '''
import json, sys
sys.path.insert(0, {repo!r})
from agent_core.scheduler import SchedulerStore

mode, database = sys.argv[1], sys.argv[2]
rest = sys.argv[3:]
store = SchedulerStore(database, delivery_lease_seconds=2)
out = {{"mode": mode}}
if mode == "route":
    now = float(rest[0])
    store.heartbeat({owner0!r}, {owner1!r}, ttl=3600, now=now)
    out["routed"] = store.route_due(now=now)
elif mode == "pending":
    out["pending"] = store.pending({owner0!r}, {owner1!r})
elif mode == "claim":
    out["claimed"] = store.claim_deliveries({owner0!r}, {owner1!r}, now=float(rest[0]))
elif mode == "complete":
    for delivery in store.claim_deliveries({owner0!r}, {owner1!r}, now=float(rest[0])):
        store.complete_delivery(delivery["id"])
        out["completed"] = delivery["id"]
elif mode == "fail":
    error, now = rest[0], float(rest[1])
    for delivery in store.claim_deliveries({owner0!r}, {owner1!r}, now=now):
        out["failed_state"] = store.fail_delivery(delivery["id"], error, now=now)
        out["delivery_id"] = delivery["id"]
elif mode == "dead":
    out["dead_letters"] = store.list_dead_letters(
        owner_session={owner0!r}, owner_agent={owner1!r}
    )
print(json.dumps(out, default=str))
'''


def _driver_path(tmp_path: Path) -> Path:
    path = tmp_path / "scheduler_driver.py"
    path.write_text(
        _DRIVER.format(repo=str(REPO_ROOT), owner0=OWNER[0], owner1=OWNER[1]), encoding="utf-8"
    )
    return path


def _run_driver(script: Path, database: Path, mode: str, *args: str) -> dict:
    completed = subprocess.run(
        [sys.executable, str(script), mode, str(database), *args],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
    )
    assert completed.returncode == 0, (
        f"driver {mode} failed rc={completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _create_job(database: Path, *, one_shot: bool) -> dict:
    store = SchedulerStore(database, delivery_lease_seconds=2)
    return store.create(
        owner_session=OWNER[0],
        owner_agent=OWNER[1],
        schedule="* * * * *",
        timezone="UTC",
        prompt="integration delivery",
        persistent=False,
        one_shot=one_shot,
        now=T0,
    )


def test_delivery_routes_completes_and_retires_one_shot(tmp_path: Path) -> None:
    script = _driver_path(tmp_path)
    database = tmp_path / "scheduler.sqlite3"
    _create_job(database, one_shot=True)

    routed = _run_driver(script, database, "route", str(T0 + 61))
    assert len(routed["routed"]) == 1
    assert routed["routed"][0]["prompt"] == "integration delivery"

    completed = _run_driver(script, database, "complete", str(T0 + 62))
    assert completed["completed"] == routed["routed"][0]["delivery_id"]

    final = _run_driver(script, database, "pending")
    assert final["pending"] == []
    # A completed one-shot retires its job entirely.
    store = SchedulerStore(database)
    assert store.list(owner_session=OWNER[0], owner_agent=OWNER[1]) == []


def test_failure_retry_backoff_and_dead_letter(tmp_path: Path) -> None:
    script = _driver_path(tmp_path)
    database = tmp_path / "scheduler.sqlite3"
    _create_job(database, one_shot=False)

    first = _run_driver(script, database, "route", str(T0 + 61))
    delivery_id = first["routed"][0]["delivery_id"]

    # Attempt 1 fails -> retry_wait with a 30s base backoff...
    attempt1 = _run_driver(script, database, "fail", "boom", str(T0 + 62))
    assert attempt1["failed_state"] == "retry_wait"
    assert attempt1["delivery_id"] == delivery_id

    # ...so a claim inside the backoff window sees nothing due.
    early = _run_driver(script, database, "claim", str(T0 + 70))
    assert early["claimed"] == []

    # Attempt 2 fails after the backoff (60s now)...
    attempt2 = _run_driver(script, database, "fail", "boom", str(T0 + 92))
    assert attempt2["failed_state"] == "retry_wait"

    # Attempt 3 fails -> max_delivery_attempts reached -> dead letter.
    attempt3 = _run_driver(script, database, "fail", "boom", str(T0 + 200))
    assert attempt3["failed_state"] == "dead_letter"

    dead = _run_driver(script, database, "dead")
    assert len(dead["dead_letters"]) == 1
    letter = dead["dead_letters"][0]
    assert letter["id"] == delivery_id
    assert letter["attempt_count"] == 3
    assert "boom" in str(letter["last_error"])


def test_expired_lease_is_recovered_by_the_next_claim(tmp_path: Path) -> None:
    script = _driver_path(tmp_path)
    database = tmp_path / "scheduler.sqlite3"
    _create_job(database, one_shot=False)

    routed = _run_driver(script, database, "route", str(T0 + 61))
    delivery_id = routed["routed"][0]["delivery_id"]

    # This process claims and dies without completing: the row stays 'running'
    # with a 2s lease, exactly like a crashed agent.
    stuck = _run_driver(script, database, "claim", str(T0 + 62))
    assert [item["id"] for item in stuck["claimed"]] == [delivery_id]

    # Still inside the lease: a fresh claim must NOT steal the delivery.
    within = _run_driver(script, database, "claim", str(T0 + 63))
    assert within["claimed"] == []

    # After expiry the delivery is failed back into the retry queue with the
    # reason recorded, and the 30s base backoff must elapse before it is
    # reclaimable — recovery never hot-loops a stuck worker.
    right_after_expiry = _run_driver(script, database, "claim", str(T0 + 65))
    assert right_after_expiry["claimed"] == []
    queued = _run_driver(script, database, "pending")
    assert [item["id"] for item in queued["pending"]] == [delivery_id]
    assert queued["pending"][0]["state"] == "retry_wait"
    assert queued["pending"][0]["last_error"] == "lease_expired"

    recovered = _run_driver(script, database, "claim", str(T0 + 96))
    assert [item["id"] for item in recovered["claimed"]] == [delivery_id]
    # Re-claiming clears the error and counts the second attempt.
    assert recovered["claimed"][0]["last_error"] is None
    assert recovered["claimed"][0]["attempt_count"] == 2


@pytest.mark.parametrize("long_database", [False, True], ids=["short-path", "long-path"])
def test_service_registration_command_contract_short_and_long_paths(
    monkeypatch, long_database: bool
) -> None:
    """The schtasks command line stays within the /TR limit for both path shapes.

    The service manager itself is stubbed (a test must never install a host
    service); what is pinned is the command contract: a short database path goes
    onto /TR directly, a long one routes through the generated .cmd launcher.
    The root is a plain mkdtemp (pytest's nested tmp_path already sits close to
    the /TR limit on some hosts, which would blur the two cases).
    """

    tmp_path = Path(tempfile.mkdtemp(prefix="svc-itg-"))
    try:
        executable = tmp_path / "python.exe"
        executable.write_bytes(b"")
        database = tmp_path / "scheduler.sqlite3"
        if long_database:
            # One long filename (not deep nesting) keeps the path creatable
            # within MAX_PATH while pushing the task command past the /TR limit.
            database = tmp_path / ("L" * 180 + ".sqlite3")
        receipt = tmp_path / "service.json"
        calls: list[list[str]] = []
        monkeypatch.setattr(scheduler_service.sys, "platform", "win32")
        monkeypatch.setattr(scheduler_service, "_run", lambda argv: calls.append(argv))
        monkeypatch.setattr(scheduler_service, "_service_is_registered", lambda value: True)

        installed = scheduler_service.install_user_service(
            executable=executable, database=database, receipt_path=receipt
        )
        create = next(call for call in calls if call[:2] == ["schtasks", "/Create"])
        task_command = create[create.index("/TR") + 1]
        assert len(task_command) <= 261
        if long_database:
            launcher = Path(installed["resources"][1])
            assert launcher.is_file()
            assert str(launcher) in task_command
            assert str(database.resolve()) in launcher.read_text(encoding="utf-8")
        else:
            assert str(database.resolve()) in task_command
            assert all(
                not str(resource).endswith(".cmd")
                for resource in installed["resources"]
            )
        scheduler_service.uninstall_user_service(
            expected_executable=executable, receipt_path=receipt
        )
        assert not receipt.exists()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
