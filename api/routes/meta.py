"""What the service is and what it was trained on."""

from __future__ import annotations

from flask import Blueprint, jsonify

from api.errors import ApiError
from api.schemas import ModelInfo
from api.service import service

meta_bp = Blueprint("meta", __name__)


@meta_bp.get("/health")
def health():
    """Liveness. Answers 200 whether or not a model is loaded.

    The container healthcheck polls this, so it reports the model separately
    rather than failing: a process serving `/docs` with no artifact is up, and
    restarting it would not produce one.
    """
    return jsonify(status="ok", model_loaded=service.available)


@meta_bp.get("/model-info")
def model_info():
    return jsonify(ModelInfo.model_validate(service.model_info()).model_dump())


@meta_bp.get("/metrics")
def metrics():
    """Full metrics payload backing the dashboard: ladder, ablations, transfer."""
    if not service.metrics:
        raise ApiError(503, "No metrics available. Run `exo train`.")
    return jsonify(service.metrics)
