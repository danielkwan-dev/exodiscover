"""The preprocessing pipeline, exposed one stage at a time."""

from __future__ import annotations

import numpy as np
from flask import Blueprint, jsonify

from api.schemas import LightCurveRequest, LightCurveStages
from api.validation import body
from exodiscover.features import lightcurve as lc

lightcurve_bp = Blueprint("lightcurve", __name__)


@lightcurve_bp.post("/lightcurve/preprocess")
@body(LightCurveRequest)
def preprocess(payload: LightCurveRequest):
    """Run the four preprocessing stages, for the 'how it works' explainer."""
    stages = lc.preprocess(np.asarray(payload.flux, dtype=float))
    flattened = stages["flattened"]
    result = LightCurveStages.model_validate(
        {
            "normalized": stages["normalized"].tolist(),
            "flattened": flattened.tolist(),
            "n_windows": int(len(stages["windows"])),
            "features": lc.window_features(flattened),
        }
    )
    return jsonify(result.model_dump())
