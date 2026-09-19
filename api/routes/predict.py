"""Scoring, and the rankings derived from it."""

from __future__ import annotations

import io

import pandas as pd
from flask import Blueprint, jsonify, request

from api.errors import ApiError
from api.limits import MAX_UPLOAD_BYTES
from api.schemas import BatchResponse, KOIInput, Prediction
from api.serialization import json_records
from api.service import service
from api.validation import body
from exodiscover.config import settings
from exodiscover.features.tabular import MULTIPLICITY_COLUMN

predict_bp = Blueprint("predict", __name__)

#: A CSV past this is refused, not truncated. The endpoint this replaced cut
#: silently at 2,000 rows and returned a partial answer that looked complete.
MAX_BATCH_ROWS = 5_000

#: Columns `build_features` cannot reconstruct a row without.
REQUIRED_CSV_COLUMNS = frozenset(
    {"koi_period", "koi_depth", "koi_duration", "koi_prad", "koi_srad", "koi_slogg"}
)


def _require_model() -> None:
    if not service.available:
        raise ApiError(503, "No trained model is loaded. Run `exo train` to produce one.")


@predict_bp.get("/discoveries")
def discoveries():
    """Top-ranked unvetted KOI candidates, with the reasons behind each."""
    path = settings.metrics_dir / "top_candidates.csv"
    if not path.exists():
        raise ApiError(503, "No discovery ranking available.")

    limit = request.args.get("limit", default=25, type=int)
    if limit is None:
        raise ApiError(422, "limit must be an integer.")

    df = pd.read_csv(path).head(max(1, min(limit, 200)))
    return jsonify(n=len(df), candidates=json_records(df))


@predict_bp.get("/skymap")
def sky_map():
    """Every KOI that can actually be placed in space, with Earth at the origin.

    Returned whole rather than filtered server-side: it is roughly 17,000 rows,
    small enough that the view filters instantly in the browser.
    """
    path = settings.metrics_dir / "sky_map.csv"
    if not path.exists():
        raise ApiError(503, "No sky map available. Run `exo ingest` then `exo skymap`.")

    df = pd.read_csv(path)
    return jsonify(n=int(len(df)), objects=json_records(df))


@predict_bp.post("/predict")
@body(KOIInput)
def predict(payload: KOIInput):
    _require_model()
    row = payload.model_dump()
    # The caller states the multiplicity directly, so hand it to the feature
    # builder under the name it expects. This used to be expressed by repeating
    # the row `count` times and letting a per-frame groupby recover the number,
    # which scored N identical rows to answer one question and relied on the
    # batch-dependent counting that build_features no longer does.
    row[MULTIPLICITY_COLUMN] = float(row.pop("n_kois_on_star", 1))

    result = service.predict_frame(pd.DataFrame([row]))[0]
    # Through the schema on the way out as well as in: `predict_frame` also
    # returns the row index, which means nothing when there is one row.
    return jsonify(Prediction.model_validate(result).model_dump())


@predict_bp.post("/predict/batch")
def predict_batch():
    _require_model()

    upload = request.files.get("file")
    if upload is None:
        raise ApiError(422, "Expected a CSV in the multipart field named 'file'.")

    contents = upload.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise ApiError(413, "File exceeds the 5 MB limit.")

    try:
        df = pd.read_csv(io.BytesIO(contents), comment="#", low_memory=False)
    except Exception as exc:
        raise ApiError(422, f"Could not parse CSV: {exc}") from exc

    if df.empty:
        raise ApiError(422, "CSV contains no rows.")
    if len(df) > MAX_BATCH_ROWS:
        raise ApiError(413, f"CSV has {len(df)} rows; the limit is {MAX_BATCH_ROWS}.")

    missing = sorted(REQUIRED_CSV_COLUMNS - set(df.columns))
    if missing:
        raise ApiError(422, f"CSV missing columns: {', '.join(missing)}")

    results = service.predict_frame(df, top_n=3)
    results.sort(key=lambda r: r["probability"], reverse=True)
    response = BatchResponse.model_validate({"n_rows": len(results), "results": results})
    return jsonify(response.model_dump())
