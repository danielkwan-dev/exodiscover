"""Blueprints, one per group of endpoints.

The split mirrors the tags the service has always documented: `meta` describes
the service and its artifacts, `predict` scores and ranks, `lightcurve` exposes
the preprocessing stages behind the explainer.
"""

from __future__ import annotations

from api.routes.lightcurve import lightcurve_bp
from api.routes.meta import meta_bp
from api.routes.predict import predict_bp

__all__ = ["lightcurve_bp", "meta_bp", "predict_bp"]
