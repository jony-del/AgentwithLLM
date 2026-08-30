# Polaris sandbox image

`Containerfile` builds the Linux guest toolchain used by every cross-kernel sandbox
invocation. Releases publish a multi-architecture OCI index for `linux/amd64` and
`linux/arm64`; the resulting index digest is injected into both
`agent_core/sandbox/sandbox-image.lock.json` and `installer/manifest.json` before the
release archives are created.

The image protocol is emitted by `/opt/polaris/bin/sandbox-probe manifest`. Runtime
selection is not complete until Polaris also passes the writable bind-mount,
read-only/non-root, and `--network none` canaries.

Python toolchain inputs are constrained by `constraints.txt`; Debian packages resolve
from a fixed snapshot, while Node, PowerShell, and Pyright versions are pinned in the
Containerfile. The root `.dockerignore` restricts
the build context to package and sandbox sources so local environments and credentials
cannot enter an image layer.
