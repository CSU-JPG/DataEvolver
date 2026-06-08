"""Common utility functions used across the DataEvolver pipeline."""

import hashlib
import pathlib


def ensure_dir(p: pathlib.Path) -> pathlib.Path:
    """Create directory if it doesn't exist and return the path."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha1_of_file(fp: pathlib.Path) -> str:
    """Compute the SHA-1 hash of a file's contents."""
    h = hashlib.sha1()
    with open(fp, "rb") as fr:
        for chunk in iter(lambda: fr.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitize_subtopic_name(name: str) -> str:
    """Normalize a subtopic name for use in file paths."""
    if not name:
        return "default"
    return name.replace("/", "_").replace("\\", "_").replace(":", "_")
