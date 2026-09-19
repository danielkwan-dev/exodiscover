"""API contract tests.

These run against a model trained on the fixture, so they exercise the real
prediction path rather than a mock.
"""

import io

import joblib
import pandas as pd
import pytest

from api.app import create_app
from api.service import band_for, service
from exodiscover.config import settings
from exodiscover.features.tabular import FEATURE_COLUMNS, build_features
from exodiscover.models.registry import candidates

VALID_KOI = {
    "koi_period": 0.837495,
    "koi_depth": 152.0,
    "koi_duration": 1.811,
    "koi_prad": 1.47,
    "koi_srad": 1.065,
    "koi_slogg": 4.35,
    "koi_steff": 5627.0,
    "koi_impact": 0.30,
    "koi_model_snr": 25.0,
    "n_kois_on_star": 2,
}


def csv_upload(df: pd.DataFrame, name: str = "koi.csv") -> dict:
    """A DataFrame as the multipart payload the batch endpoint expects."""
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return {"file": (buf, name)}


@pytest.fixture
def client(tmp_path, koi_sample, monkeypatch):
    """A client backed by a real model fitted on the fixture.

    The settings root is redirected at tmp_path before the app is built, so the
    factory's one-time `service.load()` finds this bundle rather than whatever
    sits in models/production.
    """
    monkeypatch.setattr(settings, "root", tmp_path)
    models_dir = tmp_path / "models" / "production"
    models_dir.mkdir(parents=True)

    df = koi_sample[
        koi_sample["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])
    ].reset_index(drop=True)
    X = build_features(df)
    y = (df["koi_disposition"] == "CONFIRMED").astype(int)
    model = candidates(42)["lightgbm"].fit(X, y)

    joblib.dump(
        {"model": model, "features": FEATURE_COLUMNS, "version": "test"},
        models_dir / "model.joblib",
    )

    service.__init__()
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c
    service.__init__()


def test_health_is_always_available(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_model_info_reports_the_feature_list(client):
    body = client.get("/model-info").get_json()
    assert body["available"] is True
    assert body["features"] == FEATURE_COLUMNS


def test_predict_returns_a_calibrated_probability_and_reasons(client):
    body = client.post("/predict", json=VALID_KOI).get_json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in {"planet", "false positive"}
    assert body["band"] in {"confirmed", "needs vetting", "false positive"}
    assert len(body["contributions"]) > 0
    assert {"feature", "value", "shap"} == set(body["contributions"][0])


def test_predict_rejects_a_negative_period(client):
    response = client.post("/predict", json={**VALID_KOI, "koi_period": -1.0})
    assert response.status_code == 422
    assert "koi_period" in response.text


def test_predict_rejects_a_missing_required_field(client):
    payload = {k: v for k, v in VALID_KOI.items() if k != "koi_depth"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_a_body_that_is_not_an_object(client):
    response = client.post("/predict", json=[VALID_KOI])
    assert response.status_code == 422
    assert "JSON object" in response.get_json()["detail"]


def test_batch_ranks_results_by_probability(client, koi_sample):
    body = client.post(
        "/predict/batch",
        data=csv_upload(koi_sample.head(30)),
        content_type="multipart/form-data",
    ).get_json()
    probs = [r["probability"] for r in body["results"]]
    assert probs == sorted(probs, reverse=True)
    assert body["n_rows"] == 30


def test_batch_rejects_a_csv_missing_required_columns(client):
    response = client.post(
        "/predict/batch",
        data=csv_upload(pd.DataFrame({"nonsense": [1, 2]}), "bad.csv"),
        content_type="multipart/form-data",
    )
    assert response.status_code == 422
    assert "koi_period" in response.get_json()["detail"]


def test_batch_rejects_a_request_with_no_file(client):
    response = client.post("/predict/batch", data={}, content_type="multipart/form-data")
    assert response.status_code == 422
    assert "file" in response.get_json()["detail"]


def test_lightcurve_preprocess_returns_every_stage(client):
    flux = ([1.0] * 100 + [0.99] * 10) * 4
    body = client.post("/lightcurve/preprocess", json={"flux": flux}).get_json()
    assert len(body["normalized"]) == len(flux)
    assert "num_dips" in body["features"]
    assert body["n_windows"] >= 0


def test_lightcurve_rejects_a_series_that_is_too_short(client):
    assert client.post("/lightcurve/preprocess", json={"flux": [1.0, 2.0]}).status_code == 422


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.05, "false positive"), (0.5, "needs vetting"), (0.95, "confirmed")],
)
def test_bands_partition_the_probability_range(probability, expected):
    assert band_for(probability) == expected


def test_endpoints_return_503_when_no_model_is_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    service.__init__()
    client = create_app().test_client()
    try:
        assert client.post("/predict", json=VALID_KOI).status_code == 503
        assert client.get("/metrics").status_code == 503
    finally:
        service.__init__()


@pytest.mark.parametrize(
    "origin",
    ["http://localhost:5173", "http://localhost:4173", "https://exodiscover-abc123.vercel.app"],
)
def test_cors_admits_every_origin_the_web_app_is_served_from(client, origin):
    """Dev server, preview build, and any Vercel deployment of web/."""
    response = client.get("/health", headers={"Origin": origin})
    assert response.headers.get("Access-Control-Allow-Origin") == origin


def test_cors_preflight_clears_the_json_post(client):
    """`POST /predict` sends `Content-Type: application/json`, which is not a
    CORS-simple type, so the browser sends an OPTIONS preflight first. A plain
    GET with an Origin header would not exercise this path."""
    response = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
    assert "POST" in response.headers["Access-Control-Allow-Methods"]
    assert "content-type" in response.headers["Access-Control-Allow-Headers"].lower()


def test_cors_refuses_an_unrelated_origin(client):
    response = client.get("/health", headers={"Origin": "https://not-this-project.example"})
    assert "Access-Control-Allow-Origin" not in response.headers


def test_oversized_upload_is_refused_with_a_json_413(client):
    """Werkzeug rejects this before the body is buffered, so the message has to
    come from the error handler rather than the batch route."""
    oversized = b"x" * (6 * 1024 * 1024)
    response = client.post(
        "/predict/batch",
        data={"file": (io.BytesIO(oversized), "huge.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 413
    assert response.mimetype == "application/json"
    assert "5 MB" in response.get_json()["detail"]


def test_unknown_routes_answer_with_json_not_html(client):
    """The web client reads `detail` out of every error body.

    Flask's default 404 is an HTML page, which that parse cannot survive, so
    the handler covering Werkzeug's own exceptions is part of the contract.
    """
    response = client.get("/no-such-endpoint")
    assert response.status_code == 404
    assert response.mimetype == "application/json"
    assert "detail" in response.get_json()


def test_openapi_document_is_generated_from_the_schemas(client):
    """`/docs` renders this; it has to describe the models the routes enforce."""
    spec = client.get("/openapi.json").get_json()
    assert spec["openapi"].startswith("3.1")
    assert set(spec["paths"]) >= {"/predict", "/predict/batch", "/skymap", "/health"}

    koi = spec["components"]["schemas"]["KOIInput"]
    # Generated, not transcribed: the `gt=0` on the field is what puts this here.
    assert koi["properties"]["koi_period"]["exclusiveMinimum"] == 0
    assert "koi_depth" in koi["required"]


def sky_map_fixture(tmp_path) -> None:
    """A sky map shaped like the real one: int ids, and gaps that start late.

    The gap position matters. `radius_earth` is missing for a minority of real
    rows and none of the first few, which is exactly the arrangement that hid
    this: a payload can look fine for hundreds of records and still be invalid.
    """
    metrics_dir = tmp_path / "docs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    rows = 50
    pd.DataFrame(
        {
            "name": [f"K-{i}" for i in range(rows)],
            "star_id": range(10797460, 10797460 + rows),
            "ra": [291.9] * rows,
            "dec": [48.1] * rows,
            "radius_earth": [2.1] * (rows - 10) + [None] * 10,
            "dist_pc": [None] * rows,
        }
    ).to_csv(metrics_dir / "sky_map.csv", index=False)


def test_skymap_serialises_the_integer_columns_pandas_returns(client, tmp_path):
    """`star_id` arrives from pandas as numpy.int64, which stdlib json refuses."""
    sky_map_fixture(tmp_path)

    body = client.get("/skymap").get_json()
    assert body["n"] == 50
    assert body["objects"][0]["star_id"] == 10797460
    assert isinstance(body["objects"][0]["star_id"], int)


def test_skymap_emits_no_nan_token_for_missing_measurements(client, tmp_path):
    """A gap must leave as `null`.

    Python's `json.loads` accepts the bare token `NaN`; the browser's
    `JSON.parse` does not, so asserting on the decoded body would pass while
    the page it feeds fails. The raw text is the contract.
    """
    sky_map_fixture(tmp_path)

    raw = client.get("/skymap").text
    assert "NaN" not in raw

    objects = client.get("/skymap").get_json()["objects"]
    assert objects[0]["dist_pc"] is None
    assert objects[-1]["radius_earth"] is None
    assert objects[0]["radius_earth"] == 2.1


def test_discoveries_emits_no_nan_token_for_unnamed_planets(client, tmp_path):
    """`kepler_name` is empty for every candidate that has not been confirmed."""
    metrics_dir = tmp_path / "docs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "kepoi_name": ["K00752.01", "K00753.01"],
            "kepler_name": [None, None],
            "kepid": [10797460, 10811496],
            "probability": [0.98, 0.91],
            "rank": [1, 2],
        }
    ).to_csv(metrics_dir / "top_candidates.csv", index=False)

    response = client.get("/discoveries")
    assert "NaN" not in response.text
    candidates = response.get_json()["candidates"]
    assert candidates[0]["kepler_name"] is None
    assert candidates[0]["kepid"] == 10797460
