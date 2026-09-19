"""The error contract.

Every failure leaves this service as JSON shaped ``{"detail": ...}``, including
the ones Werkzeug raises on its own: a 404, a 405, an upload past
``MAX_CONTENT_LENGTH``. Flask answers those with an HTML page by default, which
the web client cannot read -- it parses ``detail`` out of the body and falls
back to the bare status line when the parse fails, so an HTML 404 reaches the
user as "NOT FOUND" with nothing to act on.

`detail` is a string for a failure with one cause and a list of field errors
for a body that failed validation. Both are what the client already expects.
"""

from __future__ import annotations

from typing import Any

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge


class ApiError(Exception):
    """A deliberate refusal, with the status code it should leave under."""

    def __init__(self, status: int, detail: Any) -> None:
        super().__init__(detail if isinstance(detail, str) else "request rejected")
        self.status = status
        self.detail = detail


def register_error_handlers(app: Flask, *, max_upload_bytes: int) -> None:
    @app.errorhandler(ApiError)
    def _api_error(exc: ApiError):
        return jsonify(detail=exc.detail), exc.status

    @app.errorhandler(RequestEntityTooLarge)
    def _too_large(exc: RequestEntityTooLarge):
        # Werkzeug rejects the request before the body is buffered, so this is
        # the only place the limit can be reported for a genuinely huge upload.
        limit_mb = max_upload_bytes // (1024 * 1024)
        return jsonify(detail=f"File exceeds the {limit_mb} MB limit."), 413

    @app.errorhandler(HTTPException)
    def _http_error(exc: HTTPException):
        return jsonify(detail=exc.description), exc.code or 500

    @app.errorhandler(Exception)
    def _unexpected(exc: Exception):
        # Under TESTING or DEBUG, let it through. Registering a handler for
        # `Exception` at all overrides Flask's own propagation, so without this
        # a bug reaches pytest as a bare 500 with the traceback swallowed into
        # the log -- the assertion fails either way, but only one of them tells
        # you why.
        if app.testing or app.debug:
            raise exc
        # Otherwise: logged in full, reported as nothing. A traceback in the
        # response body tells a stranger the filesystem layout.
        app.logger.exception("unhandled error serving %s", exc.__class__.__name__)
        return jsonify(detail="Internal server error."), 500
