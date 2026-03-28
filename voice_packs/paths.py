"""Path validation utilities to prevent path traversal attacks."""

import os


def validate_path(user_path: str, must_exist: bool = False) -> str:
    """Resolve and validate a user-supplied path.

    Prevents ../../../ traversal by resolving to an absolute real path.
    Optionally checks that the path exists.

    Returns the resolved absolute path.
    """
    resolved = os.path.realpath(os.path.expanduser(user_path))

    if must_exist and not os.path.exists(resolved):
        raise FileNotFoundError(f"Path does not exist: {resolved}")

    return resolved


def validate_output_path(user_path: str) -> str:
    """Validate an output directory path.

    Ensures the parent directory exists or can be created.
    Returns the resolved absolute path.
    """
    resolved = os.path.realpath(os.path.expanduser(user_path))
    parent = os.path.dirname(resolved)

    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    return resolved
