"""Archive extraction for untrusted downloads (MASTER_PLAN §8: every dataset
is untrusted data).

Every archive is listed BEFORE anything is written, and refused if any member
is absolute or climbs out with `..`. Extraction goes into a new, empty
directory, and afterwards that directory is checked for symlinks. A planted
link could otherwise point a later write outside the tree.
"""
from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath


class UnsafeArchive(RuntimeError):
    pass


def check_member_names(names: list[str], archive: Path) -> None:
    for name in names:
        posix, win = PurePosixPath(name), PureWindowsPath(name)
        if posix.is_absolute() or win.is_absolute() or win.drive or ".." in posix.parts or ".." in win.parts:
            raise UnsafeArchive(f"{archive}: refusing member {name!r} (absolute or escapes the target)")


def _fresh_dir(dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        raise UnsafeArchive(f"{dest} is not empty -- every archive gets its own new directory")
    dest.mkdir(parents=True, exist_ok=True)


def _no_symlinks(dest: Path) -> None:
    links = [p for p in dest.rglob("*") if p.is_symlink()]
    if links:
        raise UnsafeArchive(f"{dest}: archive created symlinks ({links[:3]}); refusing to use it")


def list_rar(archive: Path) -> list[str]:
    out = subprocess.run(["unrar", "lb", "--", str(archive)], check=True, capture_output=True, text=True)
    return [line for line in out.stdout.splitlines() if line.strip()]


def extract_rar(archive: Path, dest: Path) -> None:
    """Also handles multi-volume sets: pass the first part (.part01.rar)."""
    check_member_names(list_rar(archive), archive)
    _fresh_dir(dest)
    subprocess.run(["unrar", "x", "-o+", "-idq", "--", str(archive), str(dest) + "/"], check=True)
    _no_symlinks(dest)


def extract_zip(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        check_member_names(zf.namelist(), archive)
        _fresh_dir(dest)
        zf.extractall(dest)
    _no_symlinks(dest)
