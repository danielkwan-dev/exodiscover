# ExoDiscover

A classifier for NASA exoplanet transit signals, trained on Kepler and tested on
TESS.

> This is a rebuilt version of a NASA Space Apps 2025 hackathon submission
> ([A World Away: Hunting for Exoplanets with
> AI](https://www.spaceappschallenge.org/2025/challenges/a-world-away-hunting-for-exoplanets-with-ai/)).
> The original reported a strong accuracy that was measuring the wrong thing:
> the Kepler table ships the vetting pipeline's own verdict as a column. This
> version quarantines those columns, holds out whole host stars rather than
> rows, and reports what the model does on a different mission's data. See
> [`docs/LEAKAGE.md`](docs/LEAKAGE.md) for the investigation.

```
77% accuracy zero-shot on TESS      (majority-class baseline 51%)
ROC-AUC 0.838 · Brier 0.177 · n = 2,562 resolved TESS objects
```

## The app

16,932 catalogued objects at their real right ascension, declination and
distance, with Earth at the origin. Drag to look, scroll to travel, click a
planet.

Both missions are shown. Kepler observed a single 22°×16° window, so its 9,444
objects form a dense beam in one direction. TESS surveyed the whole sky, so its
7,488 objects lie in every direction, the nearest 21 light years away.

![The scene](docs/screenshots/space.png)

Clicking a planet shows the model's prediction, the archive's disposition, and
the SHAP terms behind the score. Both missions are scored by the same
11-feature model, the only features the two catalogues share, so the
probabilities are comparable.

![A selected planet](docs/screenshots/space-detail.png)

## Run it

Everything runs on localhost. The trained model and its metrics are committed,
so no catalogue download or training is required.

Prerequisites: Python 3.11 or 3.12, Node 20+.

```bash
make install     # pip install -e ".[dev,api]", then npm install in web/
```

Then, in two terminals:

```bash
make serve       # Flask on :8000, interactive docs at /docs
make web         # React UI on :5173
```

Open <http://localhost:5173>.

Without `make` (Windows, or no GNU make installed):

```bash
pip install -e ".[dev,api]"
cd web && npm install && cd ..
flask --app api.wsgi run --port 8000 --reload  # terminal 1
cd web && npm run dev                          # terminal 2
```

Or run the whole stack with `docker compose up --build`.

### Regenerating the artifacts

Only needed to reproduce them rather than use the committed ones:

```bash
exo ingest              # NASA archive to data/raw/
exo train --trials 40   # ~25 min, CPU only (exo train --fast, ~6 min)
exo skymap              # joins positions and distances for the scene
```

### Checks

```bash
make test        # pytest (offline, against committed fixtures) + vitest
make lint        # ruff, mypy, eslint
```

## Layout

```
ml/exodiscover/    ingest, leakage firewall, physics features, training, evaluation
api/               Flask: typed prediction, batch CSV, SHAP, metrics, sky map
web/               React and WebGL, reading from the API
docs/              LEAKAGE.md, MODEL_CARD.md, metrics/
tests/             141 Python tests and 36 web tests, offline against fixtures
```

[`docs/LEAKAGE.md`](docs/LEAKAGE.md) covers the leakage investigation.
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) records intended use and limitations.

Python 3.11, scikit-learn, CatBoost/XGBoost/LightGBM, Optuna, SHAP, Flask,
Pydantic, gunicorn, React, TypeScript, Vite, Tailwind, pytest, vitest, ruff,
mypy, GitHub Actions. CPU only.
