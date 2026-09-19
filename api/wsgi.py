"""WSGI entrypoint.

    gunicorn api.wsgi:app          # container, and anywhere POSIX
    flask --app api.wsgi run       # development (gunicorn has no Windows port)

Kept separate from `api.app` so that importing the factory has no side effects:
the test suite needs `create_app` without a second app being built, and its
model loaded, as a consequence of the import.
"""

from __future__ import annotations

import logging

from api.app import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = create_app()
