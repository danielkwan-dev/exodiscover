# Model card — ExoDiscover binary classifier

**Version** 0.1.0 · **Family** Soft-vote ensemble (CatBoost + XGBoost +
LightGBM), isotonic calibration · **Task** Confirmed planet vs false positive,
from Kepler transit parameters.

## Start here: what it can actually do

**Trained on Kepler, tested on TESS.** On 2,562 resolved objects from a
telescope it has never seen, the model reaches **77% accuracy against a 51%
majority-class baseline**, **0.838 ROC-AUC**, and a Brier score of **0.177**.
That is the number that describes its real-world use, and it is the headline on
purpose.

The ranking survives the domain shift; the calibration does not. Measured on the
11 features both catalogs express — TESS has no equivalent for Kepler-only
quantities like signal-to-noise or KOI multiplicity.

In-domain Kepler results, and the ablations establishing that they are
leakage-free, are in the Performance section below and in `LEAKAGE.md`.

## Two models, two jobs

This project ships two artifacts. They are trained on the same Kepler rows
under the same grouped splits, and they exist because one question needs
17 features and the other needs the 11 that TESS can also express.

| | `model.joblib` | `transfer.joblib` |
|---|---|---|
| Job | Rank unvetted Kepler candidates | Score Kepler and TESS comparably |
| Features | 17 | 11 (the shared subset) |
| Family | Soft-vote ensemble, Optuna-tuned | HistGradientBoosting, untuned |
| Calibrated | Isotonic | No |
| Serves | `POST /predict`, `/discoveries` | the sky map, and the figure the app displays |
| Headline | in-domain ROC-AUC, precision@50 | **77% zero-shot on TESS** |

**The zero-shot accuracy belongs to `transfer.joblib`, not to the ensemble.**
That distinction used to be invisible: the sky map fitted its own copy of the
shared model at render time, so every probability the app displayed came from
an estimator no reported number described. The model is now persisted by
`exo train` and loaded by `exo skymap`.

Kepler objects with a resolved disposition are in `transfer.joblib`'s training
data, so the sky map scores those rows **out of fold** — each one gets its
probability from a grouped fold that never saw it. Kepler candidates and every
TESS object were never in training and are scored directly. Nothing the app
displays is a model's opinion of a row it already learned.

## Intended use

Triage. Ranking a list of unvetted Kepler transit signals so that limited
follow-up time goes to the most promising ones first. Precision at the top of
the list is what matters, and precision@50 on the held-out set is 1.000.

**Not** intended to confirm a planet. Confirmation requires radial-velocity
measurement, transit-timing analysis, or statistical validation against the
local stellar population — none of which this model performs.

## Training data

NASA Exoplanet Archive `cumulative` table (Kepler Objects of Interest).

| | |
|---|---|
| Total catalog | 9,564 KOIs across 8,214 host stars |
| Used for training | 7,585 rows with a resolved disposition (2,746 CONFIRMED, 4,839 FALSE POSITIVE) |
| Fit on | 5,287 rows / 4,647 stars (70%) |
| Held out | 2,298 rows / 1,992 stars (30%), absent from training entirely |
| Withheld entirely | 1,979 CANDIDATE rows — scored as the discovery set, never trained on |

Splits are grouped on `kepid` throughout, and hold out whole stars at the exact
requested fraction rather than the nearest `1/k` a k-fold can express. K2 is
ingested but excluded: its
archive table lacks transit depth and duration, so merging it would introduce a
missingness pattern that identifies the mission.

## Features (`model.joblib`)

17 features, listed in `ml/exodiscover/features/tabular.py::FEATURE_COLUMNS`.
Beyond the raw archive columns, three are physically derived:

- **`rho_star`** — mean stellar density from `logg` and radius.
- **`duration_ratio`** — observed transit duration over the duration implied by
  Kepler's third law at that period and stellar density. Eclipsing binaries
  deviate.
- **`depth_ratio`** — observed depth over `(Rp/R*)²`. Blends and giants deviate.
- **`n_kois_on_star`** — multiplicity, counted over the whole catalog before
  any label filtering. Multi-planet systems are rarely false positives.

Eight columns are permanently excluded as target-encoding, and three
uncertainty features were removed after measurement showed they encode
post-confirmation parameter refinement. See `LEAKAGE.md`.

## Performance (`transfer.joblib`)

Zero-shot on TESS, n = 2,562, base rate 0.495. The model was trained on Kepler
and has seen no TESS object.

| Metric | Value |
|---|---|
| **Accuracy** | **0.7697** (majority-class baseline 0.5055) |
| ROC-AUC | 0.8376 |
| PR-AUC | 0.8031 |
| Brier | 0.1771 |

```
confusion, threshold 0.5      predicted FP   predicted planet
  actual false positive             975              320
  actual planet                     270              997
```

Accuracy is quoted here and nowhere else. TESS is close to class-balanced, so
the figure carries information; Kepler's held-out slice is 63% false positives,
where predicting the majority class alone scores 0.631 and an accuracy would
flatter without informing. In-domain results are reported as ROC-AUC, PR-AUC and
Brier, which are base-rate free and therefore comparable across the two domains.

Restricted to the same 11 shared features, the in-domain reference is ROC-AUC
0.9653, PR-AUC 0.9345, Brier 0.0694 — so the cost of changing telescope is a
0.128 drop in ranking and a Brier score that more than doubles. Full in-domain
figures, confidence intervals from a host-star bootstrap, the model ladder and
the leakage ablations are all in `docs/metrics/metrics.json` and `LEAKAGE.md`.

## Calibration (`model.joblib`)

Two methods were fitted and compared on a third, star-disjoint slice that
neither the base model nor the calibrator had seen:

| Method | Brier | Distinct probabilities emitted (n = 1,053) |
|---|---|---|
| **Isotonic** (selected) | **0.0576** | **30** |
| Sigmoid (Platt) | 0.0599 | 1,053 |

Isotonic is better calibrated and was selected on that basis. But the second
column is the trade it makes: as a step function it collapses its input into 30
levels, so it is nearly useless for fine-grained ordering — many of the first 50
ranked candidates come out at exactly 1.0. Sigmoid never ties but is measurably
worse calibrated.

Rather than pick one property and lose the other, the two jobs are separated:
**the calibrated probability is what gets displayed, and the base model's raw
score is what does the ranking.** Read a displayed 1.0 as "at the top of the
range the calibration set could resolve", not as certainty.

The reliability curve is in `docs/metrics/reliability.png`. Calibration is
**Kepler-specific** and does not transfer (Brier 0.069 → 0.177 on TESS).

## Limitations

1. **Cross-mission transfer degrades.** 0.838 ROC-AUC on TESS, down from 0.965
   in-domain, with the Brier score more than doubling. The ordering is still
   useful off-domain; the probabilities are not.
2. **The training task is easier than it sounds.** Kepler false positives are
   dominated by eclipsing binaries whose implied planet radius is physically
   impossible — 99.8% of objects above 40 R⊕ are false positives. The model is
   strong at *planet vs eclipsing binary*, which is not the same as being strong
   on genuinely marginal signals. Part of the cross-mission drop is that TESS
   does not hand it the same easy separation.
3. **Candidate labels reflect follow-up selection.** Which planets got confirmed
   depends on target brightness, period, and multiplicity — not only on physics.
   This is why the model is trained on resolved dispositions only.
4. **No light-curve model.** The cached window arrays record no star identity
   and their ordering was shuffled (1,172 label runs, not the ~15 expected), so
   a leakage-free evaluation is impossible with them. Rather than publish an
   unverifiable number, the track was dropped. The preprocessing pipeline
   remains as an explainer. Fixing this needs bulk MAST downloads with identity
   preserved.
5. **Model choice is inside the noise band.** Tuning moved XGBoost by 0.0002
   PR-AUC and the top five families span 0.0030, against fold standard
   deviations near 0.006 — see the `ladder` block in `metrics.json`. Do not read
   that ordering as a finding; this problem is won by the features and the
   evaluation protocol, not by the architecture.
6. **No nested cross-validation.** The honest estimate comes from a
   star-held-out test set plus a grouped bootstrap, not from nesting the Optuna
   search inside an outer CV loop. Nesting would multiply a ~20-minute run by
   the outer fold count, which the stated CPU budget does not allow. The
   selection is therefore mildly optimistic — tuning saw the CV folds — though
   the test set stayed untouched throughout.
7. **Single archive snapshot.** Dispositions change as vetting continues; the
   model reflects the snapshot in `data/raw/`, fetched 4 October 2025. The
   shortlist has not been checked against a later snapshot.

## Reproducing

```bash
make install
exo ingest              # NASA archive -> data/raw/, query + timestamp recorded
exo train --trials 40   # ~25 min on a laptop CPU
exo train --fast        # ~6 min, skips the Optuna search
```

The trained artifact is committed, so serving the API and UI needs none of the
above — see "Run it" in the README.

Seed is 42 throughout, set once in `ml/exodiscover/config.py`.
