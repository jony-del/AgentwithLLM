"""Filesystem checks for recovery state; journal text never defines a trust root."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any


def checked_path(path: Path, root: Path) -> Path:
    """Check the lexical path and every existing component before following links."""
    path = Path(os.path.abspath(path))
    root = Path(os.path.abspath(root))
    try:
        parts = path.relative_to(root).parts
    except ValueError as exc:
        raise OSError("recovery path is outside its fixed root") from exc
    current = root
    for component in (None, *parts):
        if component is not None:
            current = current / component
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise OSError("recovery path contains a symbolic link or reparse point")
        if not stat.S_ISDIR(info.st_mode) and current != path:
            raise OSError("recovery parent is not a directory")
        if current == path and not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise OSError("recovery path is not a regular file or directory")
        if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
            raise OSError("recovery state must not be hard linked")
    if path.resolve() != path:
        raise OSError("recovery root was redirected")
    return path


def identity(path: Path) -> tuple[int, int] | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    return info.st_dev, info.st_ino


def private_path(path: Path) -> None:
    """Reject foreign ownership and write grants; never repair an existing ACL."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if os.name == "nt":
        _windows_private(path)
    elif info.st_uid != getattr(os, "getuid")() or stat.S_IMODE(info.st_mode) & 0o022:
        raise OSError("recovery state must be owned by the current user and not group/world writable")


def _windows_private(path: Path) -> None:
    """Inspect owner/DACL with Win32; SYSTEM and Administrators remain trusted."""
    import ctypes
    from ctypes import wintypes

    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    pointer = ctypes.c_void_p
    advapi.GetNamedSecurityInfoW.argtypes = [
        wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD,
        ctypes.POINTER(pointer), pointer, ctypes.POINTER(pointer), pointer, ctypes.POINTER(pointer),
    ]
    advapi.GetNamedSecurityInfoW.restype = wintypes.DWORD
    advapi.ConvertSidToStringSidW.argtypes = [pointer, ctypes.POINTER(wintypes.LPWSTR)]
    advapi.ConvertSidToStringSidW.restype = wintypes.BOOL
    advapi.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    advapi.OpenProcessToken.restype = wintypes.BOOL
    advapi.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, pointer, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    advapi.GetTokenInformation.restype = wintypes.BOOL
    advapi.GetAce.argtypes = [pointer, wintypes.DWORD, ctypes.POINTER(pointer)]
    advapi.GetAce.restype = wintypes.BOOL
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.LocalFree.argtypes = [pointer]
    kernel.LocalFree.restype = pointer

    def sid_text(sid: Any) -> str:
        value = wintypes.LPWSTR()
        if not advapi.ConvertSidToStringSidW(sid, ctypes.byref(value)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return str(value.value)
        finally:
            kernel.LocalFree(ctypes.cast(value, pointer))

    token = wintypes.HANDLE()
    if not advapi.OpenProcessToken(kernel.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        size = wintypes.DWORD()
        advapi.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))
        data = ctypes.create_string_buffer(size.value)
        if not advapi.GetTokenInformation(token, 1, data, size, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        user = sid_text(pointer.from_buffer(data))
    finally:
        kernel.CloseHandle(token)
    owner, acl, descriptor = pointer(), pointer(), pointer()
    error = advapi.GetNamedSecurityInfoW(str(path), 1, 5, ctypes.byref(owner), None, ctypes.byref(acl), None, ctypes.byref(descriptor))
    if error:
        raise ctypes.WinError(error)
    try:
        trusted = {user, "S-1-5-18", "S-1-5-32-544"}
        if not owner.value or sid_text(owner) not in trusted or not acl.value:
            raise OSError("recovery state has an unsafe Windows owner or DACL")
        # OWNER RIGHTS refers to the owner already validated above (Python's
        # Windows mode=0o700 directories use this well-known SID).
        trusted.add("S-1-3-4")
        # ACL header: BYTE revision, BYTE reserved, WORD size, WORD ace_count.
        count = ctypes.c_ushort.from_address(acl.value + 4).value
        for index in range(count):
            ace = pointer()
            if not advapi.GetAce(acl, index, ctypes.byref(ace)) or not ace.value:
                raise OSError("could not inspect recovery state DACL")
            kind = ctypes.c_ubyte.from_address(ace.value).value
            flags = ctypes.c_ubyte.from_address(ace.value + 1).value
            if flags & 0x08 or kind == 1:  # inherit-only, or access-denied ACE
                continue
            if kind != 0:  # Unknown/object/callback ACEs are not an ownership proof.
                raise OSError("unsupported recovery state DACL entry")
            mask = ctypes.c_uint32.from_address(ace.value + 4).value
            if mask & 0x500D0156 and sid_text(pointer(ace.value + 8)) not in trusted:
                raise OSError("recovery state grants write access to another Windows principal")
    finally:
        kernel.LocalFree(descriptor)
