"""The ExoDiscover Flask application.

This service began as a 20-line Flask script that ran with ``debug=True``,
loaded the model on every request, validated nothing, and silently truncated
uploads to 2,000 rows. It spent a while as a FastAPI app to escape those. It is
Flask again, and each of the four is answered here rather than avoided:

* **debug** is never enabled. Development runs `flask run --reload`, which
  reloads on edit without mounting the interactive debugger; the container runs
  gunicorn.
* **the model loads once**, in the factory below, not per request.
* **every body is validated** against a Pydantic schema by `api.validation`,
  and a rejection names the field that was wrong.
* **oversized input is refused**, with a 413 that states the limit, instead of
  being trimmed to fit.

The app is built by a factory rather than created at import so that the tests
can point `settings.root` at a temporary directory *before* the model is
loaded. Importing a ready-made app would load whatever happens to be in
`models/production` first and make the fixture a no-op.
"""

from __future__ import annotations

from flask import Flask
from flask_cors import CORS

from api.errors import register_error_handlers
from api.limits import MAX_UPLOAD_BYTES, MULTIPART_SLACK_BYTES
from api.openapi import docs_bp
from api.routes import lightcurve_bp, meta_bp, predict_bp
from api.serialization import NumpyJSONProvider
from api.service import service

#: The dev server, the preview build, and any Vercel deployment of the web app.
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:4173",
    r"https://.*\.vercel\.app",
]


def create_app() -> Flask:
    app = Flask(__name__)
    app.json = NumpyJSONProvider(app)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + MULTIPART_SLACK_BYTES

    CORS(
        app,
        resources={r"/*": {"origins": ALLOWED_ORIGINS}},
        methods=["GET", "POST"],
    )

    register_error_handlers(app, max_upload_bytes=MAX_UPLOAD_BYTES)

    for blueprint in (meta_bp, predict_bp, lightcurve_bp, docs_bp):
        app.register_blueprint(blueprint)

    service.load()
    if service.available:
        app.logger.info(
            "model %s loaded with %d features", service.version, len(service.features)
        )
    else:
        app.logger.warning("no model artifact found; prediction endpoints will return 503")

    return app
