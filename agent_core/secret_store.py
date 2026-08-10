"""Small, dependency-free adapter for the operating system's user secret store.

Plugin configuration never falls back to a plaintext credentials file.  Windows uses
Credential Manager, macOS uses Keychain, and Linux uses Secret Service through
``secret-tool``.  Callers may still store an environment-variable reference when no
system service is available.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys


class SecretStoreError(RuntimeError):
    pass


def available() -> bool:
    if os.name == "nt":
        return True
    if sys.platform == "darwin":
        return shutil.which("security") is not None
    return (
        shutil.which("secret-tool") is not None
        and bool(os.getenv("DBUS_SESSION_BUS_ADDRESS"))
    )


def put(target: str, secret: str) -> None:
    if os.name == "nt":
        _windows_put(target, secret)
        return
    if sys.platform == "darwin":
        _run(
            [
                "security", "add-generic-password", "-U", "-s", "Polaris",
                "-a", target, "-w", secret,
            ]
        )
        return
    _run(
        ["secret-tool", "store", "--label=Polaris plugin configuration", "service", "Polaris", "account", target],
        input_text=secret,
    )


def get(target: str) -> str | None:
    if os.name == "nt":
        return _windows_get(target)
    if sys.platform == "darwin":
        return _run(
            ["security", "find-generic-password", "-s", "Polaris", "-a", target, "-w"],
            missing_ok=True,
        )
    return _run(
        ["secret-tool", "lookup", "service", "Polaris", "account", target],
        missing_ok=True,
    )


def delete(target: str) -> None:
    if os.name == "nt":
        _windows_delete(target)
        return
    if sys.platform == "darwin":
        _run(
            ["security", "delete-generic-password", "-s", "Polaris", "-a", target],
            missing_ok=True,
        )
        return
    _run(
        ["secret-tool", "clear", "service", "Polaris", "account", target],
        missing_ok=True,
    )


def _run(
    command: list[str], *, input_text: str | None = None, missing_ok: bool = False
) -> str | None:
    if not command or shutil.which(command[0]) is None:
        if missing_ok:
            return None
        raise SecretStoreError("system secret store is unavailable")
    try:
        completed = subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        if missing_ok:
            return None
        raise SecretStoreError(f"system secret store failed: {type(exc).__name__}") from exc
    if completed.returncode != 0:
        if missing_ok:
            return None
        raise SecretStoreError("system secret store rejected the operation")
    return completed.stdout.rstrip("\r\n")


def _windows_put(target: str, secret: str) -> None:
    import ctypes
    from ctypes import wintypes

    class Credential(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD), ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR), ("Comment", wintypes.LPWSTR),
            ("LastWritten", wintypes.FILETIME), ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
            ("Persist", wintypes.DWORD), ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.c_void_p), ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    body = secret.encode("utf-16-le")
    if len(body) > 2560:
        raise SecretStoreError("secret exceeds Windows Credential Manager's size limit")
    blob = ctypes.create_string_buffer(body)
    credential = Credential()
    credential.Type = 1  # CRED_TYPE_GENERIC
    credential.TargetName = target
    credential.CredentialBlobSize = len(body)
    credential.CredentialBlob = ctypes.cast(blob, ctypes.POINTER(ctypes.c_ubyte))
    credential.Persist = 2  # CRED_PERSIST_LOCAL_MACHINE (user-scoped, roaming disabled)
    credential.UserName = "Polaris"
    api = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
    write = api.CredWriteW
    write.argtypes = [ctypes.POINTER(Credential), wintypes.DWORD]
    write.restype = wintypes.BOOL
    if not write(ctypes.byref(credential), 0):
        raise SecretStoreError(f"Credential Manager write failed ({ctypes.get_last_error()})")


def _windows_get(target: str) -> str | None:
    import ctypes
    from ctypes import wintypes

    class Credential(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD), ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR), ("Comment", wintypes.LPWSTR),
            ("LastWritten", wintypes.FILETIME), ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
            ("Persist", wintypes.DWORD), ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.c_void_p), ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    api = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
    pointer = ctypes.POINTER(Credential)()
    read = api.CredReadW
    read.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(ctypes.POINTER(Credential))]
    read.restype = wintypes.BOOL
    if not read(target, 1, 0, ctypes.byref(pointer)):
        error = ctypes.get_last_error()
        if error == 1168:  # ERROR_NOT_FOUND
            return None
        raise SecretStoreError(f"Credential Manager read failed ({error})")
    try:
        credential = pointer.contents
        body = ctypes.string_at(credential.CredentialBlob, credential.CredentialBlobSize)
        return body.decode("utf-16-le")
    finally:
        api.CredFree(pointer)


def _windows_delete(target: str) -> None:
    import ctypes
    from ctypes import wintypes

    api = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
    remove = api.CredDeleteW
    remove.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
    remove.restype = wintypes.BOOL
    if not remove(target, 1, 0) and ctypes.get_last_error() != 1168:
        raise SecretStoreError(
            f"Credential Manager delete failed ({ctypes.get_last_error()})"
        )
