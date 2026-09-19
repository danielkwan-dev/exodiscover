"""Request-body validation.

FastAPI derived this from the view's type annotation. Flask does not, so it is
explicit -- and being explicit is the point: there is exactly one place where a
malformed body becomes a 422 that names the field that was wrong, rather than
each view re-deriving its own idea of what "valid" means.

The original Flask endpoint this service grew out of validated nothing at all.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TypeVar

from flask import request
from pydantic import BaseModel, ValidationError

from api.errors import ApiError

M = TypeVar("M", bound=BaseModel)


def body(model: type[M]) -> Callable:
    """Validate the JSON body against `model` and pass it to the view.

    The parsed model arrives as the view's first positional argument, so the
    view never touches `request` and can be read as a pure function of its
    input.
    """

    def decorator(view: Callable) -> Callable:
        @wraps(view)
        def wrapper(*args, **kwargs):
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                raise ApiError(422, "Request body must be a JSON object.")
            try:
                parsed = model.model_validate(payload)
            except ValidationError as exc:
                raise ApiError(422, _field_errors(exc)) from exc
            return view(parsed, *args, **kwargs)

        return wrapper

    return decorator


def _field_errors(exc: ValidationError) -> list[dict]:
    """The failing fields only.

    Pydantic's own `errors()` carries the rejected input and a documentation
    URL. The input is echoed back to a caller who just sent it, and may not be
    JSON-serialisable, so only the location, the message, and the machine-
    readable type survive.
    """
    return [
        {"loc": [str(part) for part in err["loc"]], "msg": err["msg"], "type": err["type"]}
        for err in exc.errors(include_url=False)
    ]
