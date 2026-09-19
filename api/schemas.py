"""Request and response contracts.

Every field carries the unit it is measured in. The original endpoint accepted
an unvalidated CSV and silently truncated it to 2000 rows; these models make
the contract explicit and reject bad input with a 422 that says which field was
wrong.

They are load-bearing in three places, not one: `api.validation` checks request
bodies against them, the routes serialise responses back through them, and
`api.openapi` generates the published schema from them. That last one is why a
field added here needs no edit anywhere else to appear at `/docs`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class KOIInput(BaseModel):
    """One Kepler Object of Interest, in archive units."""

    koi_period: float = Field(..., gt=0, description="Orbital period, days")
    koi_depth: float = Field(..., gt=0, description="Transit depth, ppm")
    koi_duration: float = Field(..., gt=0, description="Transit duration, hours")
    koi_prad: float = Field(..., gt=0, description="Planet radius, Earth radii")
    koi_srad: float = Field(..., gt=0, description="Stellar radius, solar radii")
    koi_slogg: float = Field(..., description="Stellar surface gravity, log10 cgs")
    koi_steff: float = Field(..., gt=0, description="Stellar effective temperature, K")
    koi_impact: float = Field(0.0, ge=0, description="Impact parameter, 0 = central transit")
    koi_model_snr: float | None = Field(None, description="Transit signal-to-noise")
    koi_teq: float | None = Field(None, description="Equilibrium temperature, K")
    koi_insol: float | None = Field(None, gt=0, description="Insolation flux, Earth units")
    koi_kepmag: float | None = Field(None, description="Kepler-band magnitude")
    koi_tce_plnt_num: int | None = Field(None, ge=1, description="Planet number within its TCE")
    n_kois_on_star: int = Field(1, ge=1, description="How many KOIs share this star")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class Contribution(BaseModel):
    feature: str
    #: None when the caller omitted this input. The model scores the row
    #: regardless; inventing a number here would report a measurement that was
    #: never taken.
    value: float | None
    shap: float


class Prediction(BaseModel):
    label: str = Field(..., description="'planet' or 'false positive'")
    probability: float = Field(..., ge=0, le=1, description="Calibrated planet probability")
    band: str = Field(..., description="confirmed / needs vetting / false positive")
    contributions: list[Contribution]


class BatchRow(Prediction):
    row: int


class BatchResponse(BaseModel):
    n_rows: int
    results: list[BatchRow]


class ModelInfo(BaseModel):
    name: str
    version: str
    trained_at: str
    framing: str
    features: list[str]
    test: dict
    available: bool


class LightCurveRequest(BaseModel):
    flux: list[float] = Field(..., min_length=32, max_length=20_000)


class LightCurveStages(BaseModel):
    normalized: list[float]
    flattened: list[float]
    n_windows: int
    features: dict[str, float]
