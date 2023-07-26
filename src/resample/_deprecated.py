import warnings
from numpy import VisibleDeprecationWarning
from typing import TypeVar, Callable, Any

T = TypeVar("T")


class deprecated:
    """Deprecate function of method."""

    def __init__(self, reason: str):
        """Initialize the decorator with a reason."""
        self._reason = reason

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Wrap the target function or method."""

        def decorated_func(*args: Any, **kwargs: Any) -> T:
            warnings.warn(
                f"{func.__name__} is deprecated: {self._reason}",
                category=VisibleDeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        decorated_func.__name__ = func.__name__
        decorated_func.__doc__ = "deprecated: " + self._reason
        return decorated_func


class deprecated_parameter:
    def __init__(self, **replacements: str):
        self._replacements = replacements

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        def decorated_func(*args: Any, **kwargs: Any) -> T:
            for new, old in self._replacements.items():
                if old in kwargs:
                    warnings.warn(
                        f"keyword {old!r} is deprecated, please use {new!r}",
                        category=VisibleDeprecationWarning,
                        stacklevel=2,
                    )
                    kwargs[new] = kwargs[old]
                    del kwargs[old]
            return func(*args, **kwargs)

        decorated_func.__name__ = func.__name__
        return decorated_func
