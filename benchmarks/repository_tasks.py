"""Evaluate fixed repository tasks offline; input files are never executed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_core.benchmarks.repository import compare, evaluate


def read(path: Path) -> dict:
    with path.open("rb") as handle:
        raw = handle.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("evaluation input exceeds 8 MiB")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("evaluation input must be an object")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--treatment", default="",
                        help="comma-separated environment keys allowed to differ, e.g. model,hardware")
    parser.add_argument("--unpaired", action="store_true",
                        help="aggregate rate deltas only, without per-task pairing")
    args = parser.parse_args()
    report = evaluate(read(args.manifest), read(args.observations), k=args.k)
    if args.before:
        treatment = tuple(key.strip() for key in args.treatment.split(",") if key.strip())
        report["comparison"] = compare(evaluate(read(args.manifest), read(args.before), k=args.k), report,
                                       treatment=treatment, paired=not args.unpaired)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
