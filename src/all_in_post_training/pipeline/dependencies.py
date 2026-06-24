from __future__ import annotations

import importlib.metadata
import importlib.util
from typing import Any


class MissingOptionalDependencyError(RuntimeError):
    """Raised when a requested optional training dependency is unavailable."""


def optional_dependency_status(package_name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return {"available": False, "package": package_name, "version": None}
    try:
        version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return {"available": True, "package": package_name, "version": version}


def require_optional_dependency(package_name: str, purpose: str) -> dict[str, Any]:
    status = optional_dependency_status(package_name)
    if not status["available"]:
        raise MissingOptionalDependencyError(
            f"{purpose} requires optional package {package_name!r}; "
            "install it or run without the require flag"
        )
    return status
