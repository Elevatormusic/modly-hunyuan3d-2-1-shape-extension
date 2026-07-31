"""
Integrity guard for the shape checkpoint.

`model.fp16.ckpt` is a torch ZIP archive. When its 7.4 GB download is cut
short — a closed app, a dropped connection, a full drive, an antivirus
quarantine, a `git clone` without git-lfs — torch.load fails deep inside the
container reader with

    RuntimeError: PytorchStreamReader failed reading zip archive:
                  failed finding central directory

which names neither the file nor the cause. Worse, the download gate used to
ask only whether the path existed, so a damaged checkpoint was permanently
sticky: every later launch reused it and reproduced the same traceback, with
no way out short of deleting the file by hand.

The helpers here make that damage cheap to detect (a size floor plus a scan of
the file's tail for the ZIP end-of-central-directory record — never a 7 GB
read) and repairable (drop the file *and* the huggingface_hub sidecars that
would otherwise let snapshot_download consider it already fetched).
"""
from __future__ import annotations

import os
from pathlib import Path

# The published tencent/Hunyuan3D-2.1 checkpoint is 7,366,389,768 bytes. The
# floor sits below that with room for an upstream re-upload; it only has to
# separate a real checkpoint from a stub (a git-lfs pointer is ~130 bytes) or
# a partial write.
MIN_CKPT_BYTES = 7_000_000_000

_EOCD_SIG = b"PK\x05\x06"
# The end-of-central-directory record is 22 bytes plus a comment of at most
# 65535, so it always begins within this many bytes of the end of the file.
_EOCD_TAIL = 22 + 0xFFFF

# Substrings that mark a container-level read failure — a damaged file — as
# opposed to a missing dependency, a bad config key, or an out-of-memory.
_CORRUPT_MARKERS = (
    "central directory",
    "pytorchstreamreader",
    # torch's ZIP container reader. Everything it raises — malformed layout,
    # bad offsets, CRC mismatch — means the archive cannot be read, so the
    # file itself is the problem. Found on-device: a checkpoint replaced by
    # an unrelated ZIP raises only "[enforce fail at inline_container.cc:176]
    # . file in archive is not in a subdirectory", matching none of the rest.
    "inline_container",
    "not a zip file",
    "invalid load key",
    "unexpected eof",
    "truncated",
)


def has_zip_footer(path, tail: int = _EOCD_TAIL) -> bool:
    """True if the file ends with a ZIP end-of-central-directory record.

    Reads at most `tail` bytes from the end, so this stays O(1) on a 7 GB
    checkpoint. A truncated download loses the footer, because the central
    directory is written last.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            fh.seek(max(0, size - tail))
            return _EOCD_SIG in fh.read()
    except OSError:
        return False


def ckpt_ok(path, min_bytes: int = None) -> bool:
    """True if `path` looks like a complete torch checkpoint.

    `min_bytes` defaults to MIN_CKPT_BYTES, read at call time so the constant
    stays the single source of truth.
    """
    floor = MIN_CKPT_BYTES if min_bytes is None else min_bytes
    try:
        if not os.path.isfile(path) or os.path.getsize(path) < floor:
            return False
    except OSError:
        return False
    return has_zip_footer(path)


def size_note(path) -> str:
    """Human-readable size for an error message; tolerates a missing file."""
    try:
        n = os.path.getsize(path)
    except OSError:
        return "file missing"
    return f"{n:,} bytes, expected at least {MIN_CKPT_BYTES:,}"


def looks_corrupt(exc: BaseException) -> bool:
    """True if `exc` reads as a damaged-file failure rather than a real bug."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _CORRUPT_MARKERS)


def purge(local_dir, rel_path) -> list:
    """Delete a downloaded file and the huggingface_hub sidecars recording it.

    snapshot_download keeps an etag next to each file it fetched under
    `<local_dir>/.cache/huggingface/download/`; leaving those behind lets a
    re-download decide the corrupt file is already present. Returns the paths
    removed. Never raises — this runs on a recovery path.
    """
    root = Path(local_dir)
    rel = Path(rel_path)
    removed = []

    target = root / rel
    try:
        if target.is_file():
            target.unlink()
            removed.append(str(target))
    except OSError:
        pass

    sidecars = root / ".cache" / "huggingface" / "download" / rel.parent
    try:
        for path in sorted(sidecars.glob(rel.name + ".*")):
            try:
                path.unlink()
                removed.append(str(path))
            except OSError:
                pass
    except OSError:
        pass

    return removed
