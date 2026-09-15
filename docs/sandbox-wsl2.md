# Windows WSL2 OCI sandbox

Polaris runs sandboxed tools in a Linux container managed by Podman Machine on WSL2.
Activating the Python `.venv` does not start this virtual machine.

## Startup

To opt in for this checkout, add the following to `agent.local.toml`:

```toml
[sandbox]
enabled = true

[sandbox.container]
runtime = "podman"
auto_start_machine = true
podman_machine_name = "podman-machine-default"
auto_pull = false
```

`auto_start_machine` defaults to `false`. When enabled, Polaris first checks
`podman info`. If unavailable, it verifies that the named, existing WSL2 machine
matches the active Podman connection, starts a stopped machine, and waits for the
runtime. Discovery, startup and readiness share a 120-second deadline. A machine
already starting is waited on; it is not started again. Failures retain exit codes
and bounded diagnostic output. Startup progress goes to stderr.

Polaris does not initialize/reset machines, change the default connection, enable
Windows components, or configure login startup. It leaves the machine running on
exit. Automatic startup applies only to the Windows Podman container backend.
Repository-controlled startup settings require the existing repository trust
approval; user/local overrides follow the existing configuration precedence.
`CONTAINER_CONNECTION` is respected; an unavailable `CONTAINER_HOST` override is
reported without starting a local machine. `polaris health` does not start machines.

With automatic startup disabled, start the machine explicitly:

```powershell
podman machine list
podman system connection list
podman machine start podman-machine-default
podman info
```

If no machine exists, initialize one explicitly using the installer or
`podman machine init podman-machine-default`. If the connection does not match,
inspect the connection list and the configured machine name before changing either.
A running machine with a broken socket is reported as a connection failure, not
as a stopped machine.

## Image and isolation verification

Release installations use the repository digest in
`agent_core/sandbox/sandbox-image.lock.json`. The installer prepares it. For an
existing developer install, pull this exact release image once if missing:

```powershell
$sandboxImage = (Get-Content -Raw agent_core/sandbox/sandbox-image.lock.json | ConvertFrom-Json).image
podman pull $sandboxImage
polaris run "Say hello without tools" --provider fake --sandbox
polaris health --provider fake --profile runtime
```

Automatic machine startup and image pulling are independent settings. Keep
`auto_pull = false` to prevent ordinary startup from downloading images. Registry
authentication, missing digests, and network failures must be resolved before
preparation can succeed; a mutable tag is not a substitute for the locked image.

### Build a local image

From an activated development environment with `uv`, Podman and the project's
`all,dev` extras installed, run at the repository root:

```powershell
python tools/build_sandbox.py
polaris run "Say hello without tools" --provider fake --sandbox --no-memory
polaris health --provider fake --profile runtime --no-memory --json
```

The build script exports constraints from the existing `uv.lock` with
`uv export --locked --extra all --extra dev --no-emit-project --no-hashes`, retaining
platform markers and all feature/development dependencies. It builds only
`linux/amd64`, explicitly supplies `TARGETARCH=amd64`, and uses the existing
`.dockerignore` allowlist. Host environments, credentials, `.env`, and runtime
data are excluded. Node and PowerShell downloads are checksum-verified.

The pinned `2026.7.10` Git/Fetch/Time reference servers still require the MCP 1.x
server API, while Polaris uses SDK 2.x. They therefore run in the independent
`/opt/polaris/mcp` environment, locked by `sandbox/mcp-servers/uv.lock` with SDK
`1.28.1`. The root lock and the main Python environment retain SDK `2.1.1`.
The guest manifest declares `mcp-python`; configured `python -m mcp_server_*`
commands for those three servers use it. Both environments must pass `pip check`.
No host dependency downgrade or compatibility patch to third-party code is needed.

The actual local image ID comes from Podman's
[`--iidfile`](https://docs.podman.io/en/latest/markdown/podman-build.1.html#iidfile-imageidfile).
Before activation, the script verifies the complete ID, guest protocol/toolchain,
workspace mount, security/network canaries, `pip check`, and the real OCI integration
suite (including MCP, LSP, Node and Unicode paths). Only after these pass does it
update `agent.local.toml`. The previous file and acceptance record are saved under
`tmp/sandbox-build-*`. Failures leave the original local configuration in place;
`--no-activate` accepts a candidate without changing that file.

Local configuration uses `image = "sha256:<64 lowercase hex digits>"`, explicit
`runtime = "podman"`, and `auto_pull = false`. Startup settings are preserved.
Local IDs are inspected in full and every probe/tool run uses `--pull=never`.
Missing images require rebuilding; shortened IDs and mutable tags are rejected.
Local IDs are never written to the release lock or installer manifest and no image
is published. To restore a previous selection, copy the saved
`agent.local.toml.before` over `agent.local.toml`.

Health checks use the project's resolved sandbox settings, including local image
overrides. They disable automatic startup and pulling on a copy of that config,
so checking health neither wakes a stopped machine nor downloads an image.
OCI runs keep stdin open for MCP/LSP/workflow protocols. Windows OCI clients retain
their application-data and selected-connection environment variables to find the
same Podman connection; these host settings are not injected into the Linux guest.

If an external hook points to a missing Python script, Polaris treats that as a
launch failure and applies the hook's configured `fail_mode`. The example
`./.polaris/hooks/validate_prompt.py` in `agent.toml` uses the default `open` mode,
so its absence is logged. A `closed` hook still blocks when missing, and a real
hook that exits with code 2 still blocks in either mode. The built-in prompt
validation firewall remains enabled.

Runtime readiness alone does not enable sandbox execution. Polaris also verifies
the guest protocol and toolchain, a writable workspace bind mount, a non-root user,
read-only root filesystem and denied network access. A local path such as
`E:\project` maps to `/mnt/e/project` in the WSL2 guest. UNC/network workspaces are
unsupported. Containers drop all capabilities, disallow new privileges, and use
`--network none`; other host directories are not mounted unless explicitly allowed.

Any failed preparation leaves the sandbox unavailable and prevents tool execution
on the host. Successful preparation reports `prepared=true`, `effective=true`,
the selected runtime and immutable image in the health sandbox canary.
