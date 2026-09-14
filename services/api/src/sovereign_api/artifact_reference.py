"""Portable, bounded relative identifiers for written artifacts."""

import unicodedata


MAX_ARTIFACT_REFERENCE_BYTES = 1_024


def valid_artifact_reference(value: object) -> bool:
    """Accept exact POSIX-style relative paths without normalization or aliases."""

    if type(value) is not str or not value or value != value.strip():
        return False
    if value.startswith("/") or "\\" in value or ":" in value:
        return False
    if any(unicodedata.category(character) == "Cc" for character in value):
        return False
    if any(part in ("", ".", "..") for part in value.split("/")):
        return False
    try:
        return len(value.encode("utf-8", errors="strict")) <= MAX_ARTIFACT_REFERENCE_BYTES
    except UnicodeEncodeError:
        return False
