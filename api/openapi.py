"""The OpenAPI document, and the Swagger page that renders it.

FastAPI produced both as a side effect of the type annotations. Here the
component schemas are still generated -- from the same Pydantic models the
routes validate against -- so the half of the document that describes payloads
cannot drift from the code: add a field to `api.schemas` and it appears at
`/openapi.json` without anyone editing this file. Only the paths, which no
amount of introspection can infer, are written by hand.
"""

from __future__ import annotations

from typing import Any

from flask import Blueprint, Response, jsonify
from pydantic.json_schema import models_json_schema

from api.limits import MAX_UPLOAD_BYTES
from api.routes.predict import MAX_BATCH_ROWS
from api.schemas import (
    BatchResponse,
    KOIInput,
    LightCurveRequest,
    LightCurveStages,
    ModelInfo,
    Prediction,
)

docs_bp = Blueprint("docs", __name__)

API_TITLE = "ExoDiscover API"
API_VERSION = "0.1.0"
API_DESCRIPTION = (
    "Calibrated exoplanet classification over the NASA Kepler KOI catalog. "
    "Every probability is isotonic-calibrated and every prediction carries "
    "its SHAP contributions."
)

_DOCUMENTED_MODELS = (
    KOIInput,
    Prediction,
    BatchResponse,
    ModelInfo,
    LightCurveRequest,
    LightCurveStages,
)


def _component_schemas() -> dict[str, Any]:
    """Every model, and every model they nest, as OpenAPI components.

    `models_json_schema` is the multi-model entry point: it resolves shared
    definitions once, so `Contribution` appears a single time even though both
    `Prediction` and `BatchResponse` reach it.
    """
    _, definitions = models_json_schema(
        [(model, "validation") for model in _DOCUMENTED_MODELS],
        ref_template="#/components/schemas/{model}",
    )
    return definitions.get("$defs", {})


def _ref(model: type) -> dict:
    return {"$ref": f"#/components/schemas/{model.__name__}"}


def _json_response(description: str, model: type | None = None) -> dict:
    block: dict[str, Any] = {"description": description}
    if model is not None:
        block["content"] = {"application/json": {"schema": _ref(model)}}
    return block


def _json_request(model: type) -> dict:
    return {
        "required": True,
        "content": {"application/json": {"schema": _ref(model)}},
    }


_UNAVAILABLE = _json_response("A required artifact has not been produced yet.")
_INVALID = _json_response("The request body failed validation.")


def spec() -> dict[str, Any]:
    """The OpenAPI 3.1 document for this service."""
    return {
        "openapi": "3.1.0",
        "info": {
            "title": API_TITLE,
            "version": API_VERSION,
            "description": API_DESCRIPTION,
        },
        "tags": [
            {"name": "meta", "description": "What the service is and what it was trained on."},
            {"name": "predict", "description": "Scoring, and the rankings derived from it."},
            {"name": "lightcurve", "description": "The preprocessing stages, one at a time."},
        ],
        "components": {"schemas": _component_schemas()},
        "paths": {
            "/health": {
                "get": {
                    "tags": ["meta"],
                    "summary": "Liveness, and whether a model is loaded.",
                    "responses": {"200": _json_response("The service is up.")},
                }
            },
            "/model-info": {
                "get": {
                    "tags": ["meta"],
                    "summary": "The loaded model, its version, and its feature list.",
                    "responses": {"200": _json_response("Model metadata.", ModelInfo)},
                }
            },
            "/metrics": {
                "get": {
                    "tags": ["meta"],
                    "summary": "Full metrics payload: ladder, ablations, transfer.",
                    "responses": {
                        "200": _json_response("The metrics written by `exo train`."),
                        "503": _UNAVAILABLE,
                    },
                }
            },
            "/discoveries": {
                "get": {
                    "tags": ["predict"],
                    "summary": "Top-ranked unvetted candidates, with their reasons.",
                    "parameters": [
                        {
                            "name": "limit",
                            "in": "query",
                            "required": False,
                            "schema": {
                                "type": "integer",
                                "default": 25,
                                "minimum": 1,
                                "maximum": 200,
                            },
                        }
                    ],
                    "responses": {
                        "200": _json_response("The ranked candidates."),
                        "503": _UNAVAILABLE,
                    },
                }
            },
            "/skymap": {
                "get": {
                    "tags": ["predict"],
                    "summary": "Every KOI that can be placed in space, Earth at the origin.",
                    "responses": {
                        "200": _json_response("The full sky map, unfiltered."),
                        "503": _UNAVAILABLE,
                    },
                }
            },
            "/predict": {
                "post": {
                    "tags": ["predict"],
                    "summary": "Score one KOI.",
                    "requestBody": _json_request(KOIInput),
                    "responses": {
                        "200": _json_response(
                            "A calibrated probability and its reasons.", Prediction
                        ),
                        "422": _INVALID,
                        "503": _json_response("No trained model is loaded."),
                    },
                }
            },
            "/predict/batch": {
                "post": {
                    "tags": ["predict"],
                    "summary": f"Score a CSV of up to {MAX_BATCH_ROWS:,} rows, ranked.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "required": ["file"],
                                    "properties": {
                                        "file": {
                                            "type": "string",
                                            "format": "binary",
                                            "description": (
                                                "KOI catalog CSV, at most "
                                                f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
                                            ),
                                        }
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": _json_response("Every row scored, highest first.", BatchResponse),
                        "413": _json_response("The upload is past a size or row limit."),
                        "422": _json_response("The CSV could not be parsed or lacks columns."),
                        "503": _json_response("No trained model is loaded."),
                    },
                }
            },
            "/lightcurve/preprocess": {
                "post": {
                    "tags": ["lightcurve"],
                    "summary": "Normalise, flatten, and window a flux series.",
                    "requestBody": _json_request(LightCurveRequest),
                    "responses": {
                        "200": _json_response("Each stage of the pipeline.", LightCurveStages),
                        "422": _INVALID,
                    },
                }
            },
        },
    }


@docs_bp.get("/openapi.json")
def openapi_document():
    return jsonify(spec())


#: Swagger UI is loaded from a CDN rather than vendored: it is a development
#: convenience, and shipping ~3 MB of bundled JavaScript into the container to
#: serve it would be paying for the convenience in image size.
_SWAGGER_PAGE = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{API_TITLE}</title>
    <link
      rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui.css"
    />
  </head>
  <body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui-bundle.js"></script>
    <script>
      window.ui = SwaggerUIBundle({{
        url: "/openapi.json",
        dom_id: "#swagger-ui",
        deepLinking: true,
      }});
    </script>
  </body>
</html>
"""


@docs_bp.get("/docs")
def swagger_ui():
    return Response(_SWAGGER_PAGE, mimetype="text/html")
