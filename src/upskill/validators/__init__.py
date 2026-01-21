"""Custom validators for upskill test cases."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from upskill.models import ValidationResult

# Type alias for validator functions
ValidatorFunc = Callable[..., "ValidationResult"]

# Registry of validators
_VALIDATORS: dict[str, ValidatorFunc] = {}


def register_validator(name: str) -> Callable[[ValidatorFunc], ValidatorFunc]:
    """Decorator to register a validator function.

    Usage:
        @register_validator("my_validator")
        def my_validator(workspace: Path, output_file: str, **config) -> ValidationResult:
            ...
    """

    def decorator(func: ValidatorFunc) -> ValidatorFunc:
        _VALIDATORS[name] = func
        return func

    return decorator


def get_validator(name: str) -> ValidatorFunc | None:
    """Get a validator by name.

    Args:
        name: The validator name (e.g., "hf_eval_yaml")

    Returns:
        The validator function, or None if not found
    """
    return _VALIDATORS.get(name)


def list_validators() -> list[str]:
    """List all registered validator names."""
    return list(_VALIDATORS.keys())
