# Leakage investigation

What was wrong with the original pipeline, what it cost, and what the honest
numbers are. Every figure here is reproduced by `exo train`, which writes
`docs/metrics/metrics.json`.

Archive snapshot: Kepler cumulative KOI table, 9,564 rows / 8,214 host stars.

---

## 1. Label leakage — the large one

`data/raw/koi.csv` ships five columns that encode the answer:

| Column | What it actually is |
|---|---|
| `koi_score` | The Robovetter's disposition confidence — its verdict, as a number |
| `koi_fpflag_nt` | "Not transit-like" vetting flag |
| `koi_fpflag_ss` | "Stellar eclipse" vetting flag |
| `koi_fpflag_co` | "Centroid offset" vetting flag |
| `koi_fpflag_ec` | "Ephemeris match / contamination" flag |

A model given these does not learn transit physics; it learns to read the
vetting pipeline's mind. The original `predict.py` and `train_xgboost.py`
selected feature columns by hand and never excluded them.

**Effect: ROC-AUC 0.9999 with them, 0.9840 without.** The leaky setup is
essentially a lookup table.

These are now quarantined in `ml/exodiscover/data/schema.py::LEAKY_COLUMNS`,
dropped on the way into `build_features` and asserted absent on the way out.
`tests/test_schema.py` and `tests/test_features.py` fail the build if one ever
reaches the feature matrix.

### Two guards, because a name check is not enough

`assert_no_leakage` compares column *names*. That catches the original defect —
hand-picked feature lists containing `koi_score` — and nothing subtler. It
cannot see a leaky column that arrives under a different name, and since
`build_features` restricts its output to `FEATURE_COLUMNS`, it can only ever
fire there if that list itself is edited. A guard that can only catch the
mistake you already know about is worth stating honestly rather than trusting.

So a second guard reads the *values*. `find_derived_leakage` measures rank
correlation between every shipped feature and every Robovetter column, and the
threshold comes from measurement rather than taste:

| | \|Spearman\| vs `koi_score` |
|---|---|
| Strongest *legitimate* pairing (`log_prad` vs `koi_fpflag_ss`) | 0.553 |
| **Threshold** | **0.80** |
| `koi_score` shipped under another name | 1.000 |
| `koi_score` halved and mixed with noise | 0.885 |

That 0.553 is not leakage: eclipsing binaries are both large and flagged as
such, so the physics and the vetting flag agree without either causing the
other. The threshold sits clear above it and below a diluted leak.

Spearman rather than Pearson, deliberately — a leak reintroduced through a log,
a rescaling, or a rank is still a leak, and rank correlation is blind to the
transform in a way Pearson is not. It runs once per training run against the
raw catalog, not on the serving path, where it would cost a correlation per
feature per request and need many rows to mean anything.

A third guard runs in CI, and it deliberately does **not** test a score against
a fixed threshold — a clean number can legitimately be high, so a ceiling would
fire on honest runs and stay silent on subtle leaks. It checks the shape of the
table above instead: whether the clean pipeline has closed the gap on the
deliberately-leaky one. That is what leakage would actually look like.

## 2. Split leakage — smaller than expected

Kepler lists 9,564 KOIs across only 8,214 stars, up to seven on one star.
Sibling KOIs share stellar parameters and photometry, so a random row split
puts related rows on both sides. `train_xgboost.py:110` split on rows; worse,
the light-curve pipeline split on 256-point *windows* cut from the same
continuous series.

**I expected this to be a major inflation source. It is not.**

| Split | ROC-AUC |
|---|---|
| Random rows | 0.9846 |
| Grouped by host star | 0.9840 |

The difference is within noise, and grouping scores marginally *lower* — that
is, the contamination it removes was worth 0.0006. The reason is arithmetic:
the binary training set holds 7,585 rows across 6,639 stars, so only about 21%
of rows have a sibling anywhere in the data, and most of those siblings are
pairs. There is not enough overlap for the contamination to matter.

Grouping is kept because it is the methodologically correct choice and it costs
nothing — but the honest finding is that on this dataset it was not the problem.
It would be the problem on the light-curve windows, where ~1,800 windows come
from a handful of stars (see §4).

## 3. Temporal leakage — the subtle one

Three engineered features were in the original design and have been **removed
after measurement**: `rel_err_period`, `rel_err_depth`, `rel_err_duration`
(relative parameter uncertainty, `err / value`).

They look like legitimate signal — a noisy measurement should be less
trustworthy. The Confirmed-vs-Candidate diagnostic showed what they really
encode:

| Feature | mean abs SHAP | Category |
|---|---|---|
| `rel_err_depth` | 0.930 | uncertainty |
| `koi_model_snr` | 0.769 | transit physics |
| `n_kois_on_star` | 0.614 | selection |
| `rel_err_period` | 0.454 | uncertainty |
| `log_period` | 0.391 | selection |
| `rel_err_duration` | 0.369 | uncertainty |

Selection proxies totalled 1.005 and transit physics 1.037 — near-tied — but
the three uncertainty terms together reached **1.75**, the largest group by a
wide margin.

The mechanism is temporal. A planet becomes CONFIRMED through follow-up
observation, and *that same follow-up refines its published parameters*. The
uncertainty is therefore a consequence of the label rather than a property of
the signal, and it would not be available at the moment a real classifier has
to make its call.

Removing them cost the binary model **0.003 ROC-AUC (0.9865 → 0.9835 as
measured at the time of the decision, on the then-current pipeline)**. Cheap
enough that keeping a feature I cannot defend was
not worth it. `tests/test_features.py::test_uncertainty_features_stay_excluded`
locks the decision in.

## 4. The light-curve track cannot be evaluated

The original CNN reported strong accuracy on 1,811 windows. Two defects make
that number unrecoverable:

1. `create_training_data.py:21-23` selected targets with `.head(5)` per class —
   at most **15 stars** total, each contributing up to 100 overlapping windows.
2. Windows were split as independent rows, so overlapping segments of the same
   star's photometry appeared in both train and test.

The obvious fix is a star-grouped split. **It is not possible with the cached
data.** The arrays record no star identity, and reconstructing it from the label
ordering fails: the labels form **1,172 runs**, not the ~15 expected if each
target's windows were contiguous. The ordering was shuffled before saving, so
provenance is gone.

Rather than fabricate star IDs to produce a number, no light-curve model ships.
The preprocessing pipeline is retained as an explainer
(`ml/exodiscover/features/lightcurve.py`, tested offline). Fixing this properly
requires re-downloading light curves from MAST with identity preserved, which is
outside this project's stated CPU budget.

## 5. Cross-mission generalisation — the result

Trained on Kepler, evaluated zero-shot on 2,562 resolved TESS objects using
only features both catalogs express:

| | n | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| Kepler (in-domain) | 2,298 | 0.9653 | 0.9345 | 0.0694 |
| TESS (zero-shot) | 2,562 | **0.8376** | 0.8031 | **0.1771** |

**Zero-shot accuracy on TESS is 0.7697, against a majority-class baseline of
0.5055.** That is the one accuracy this project quotes, because it is the one
measured on a near-balanced population where the number carries information.
Accuracy is not used for the in-domain comparison: Kepler's held-out slice is
63% false positives, so the baseline alone is 0.631 and the two figures would
not be comparable — the gap between them would be part real degradation and
part arithmetic. ROC-AUC is base-rate free, so the comparison uses it.

Measured that way, performance falls by 0.128. The Brier score
more than doubles. **The ranking largely survives the domain shift; the
calibration does not** — probabilities that are trustworthy on Kepler are not
trustworthy on TESS. TESS has shorter baselines, a redder bandpass, larger
pixels and therefore more blending, and a different false-positive population.

This figure also measures one of the fixes above. TESS does not carry several
KOI columns at all, and an earlier feature builder filled those gaps with the
*Kepler* median — a fabricated value asserting a measurement TESS never made.
Letting them stay missing, for the boosted models to route down a learned
branch, moved zero-shot ROC-AUC from 0.7619 to 0.8376 with no change to the
model itself. Imputing across a domain boundary was quietly costing 0.076.

This is the number that says what the model can actually do on data this project
did not train on, which is why it is the one reported.

---

## 6. How certain are these numbers?

**Confidence intervals** come from a bootstrap that resamples **host stars, not
rows**, for the same reason the splits are grouped: sibling KOIs share stellar
parameters and are not independent draws, so a row bootstrap reports an interval
narrower than the data supports. Intervals for every reported metric are in
`docs/metrics/metrics.json`.

**Calibration involved a real trade**, not a default. Isotonic and sigmoid were
both fitted and scored on a third, star-disjoint slice:

| Method | Brier | Distinct probabilities (n = 1,053) |
|---|---|---|
| Isotonic (selected) | 0.0576 | 30 |
| Sigmoid | 0.0599 | 1,053 |

Isotonic calibrates better; sigmoid never ties. Isotonic's 30 levels mean many
of the top 50 candidates land on exactly 1.0, which is useless for ordering. So
the two jobs are split: the calibrated probability is displayed, the raw model
score does the ranking.

**Not done: nested cross-validation.** The Optuna search scores against the
same grouped folds that rank the ladder, so the *cross-validated* figures for
the selected model are mildly optimistic. Nesting the search inside an outer
loop would multiply a 20-minute run by the outer fold count, which is beyond
the CPU budget. What the nesting would protect — the reported held-out numbers
— is protected instead by the order of operations: the split happens first, and
selection and tuning see only the training stars.

### That order was wrong until recently

This section previously claimed the held-out set was never touched by tuning.
It was. `fit_candidates` and `tune_best` both ran on the full feature matrix,
and `grouped_train_test_split` carved the test set out of it *afterwards* — so
every held-out star had already sat inside the folds that chose the model and
its hyperparameters — in the one document that exists to catch exactly this.

**Corrected and re-measured: it was worth 0.0000.** Every reported figure is
identical to four decimal places before and after the fix.

| | leaked selection | split first |
|---|---|---|
| In-domain ROC-AUC | 0.9839 | 0.9839 |
| In-domain PR-AUC | 0.9666 | 0.9666 |
| In-domain Brier | 0.0440 | 0.0440 |
| precision@50 | 1.000 | 1.000 |

The reason is the same arithmetic that made split leakage a non-event in §2.
`soft_vote` tops the ladder by 0.0006 PR-AUC over `xgboost`, against a
fold-to-fold spread of 0.0049 — the families are too close for the choice
between them to carry information about any particular fold. And the winner is
an ensemble, which this project does not tune, so the selected estimator had no
searched hyperparameters for a leak to bite on. Selection had nothing to
overfit *to*.

That is the honest finding, and it is not the same as the defect being
harmless. The measurement is a property of this ladder: if one family had won
by a clear margin, or if a tuned family had taken the top slot, the same code
path would have produced an inflated number with nothing to signal it.

It is worth stating plainly rather than quietly correcting: a name-based guard
and a value-based guard both passed, because neither looks at *when* a row is
used. Leakage through the order of operations is invisible to a check on the
feature matrix.

The split now runs before anything that makes a choice, and two tests fail the
build if that is ever reversed —
`test_the_ladder_never_sees_a_held_out_star` and
`test_tuning_never_sees_a_held_out_star` in `tests/test_cli.py`, which compare
the stars handed to model selection against the stars the reported metrics are
computed on.

**The zero-shot transfer number was never affected.** `run_transfer` fits its
own model on its own grouped split with no ladder and no Optuna, so the 77% on
TESS is unchanged by this correction.

## Summary

| Issue | Where | Effect |
|---|---|---|
| Robovetter verdict columns as features | `predict.py`, `train_xgboost.py` | 0.9999 → 0.9840 |
| Row-level rather than star-level splits | `train_xgboost.py:110` | negligible here; fatal for light curves |
| Uncertainty features encoding follow-up | new design, caught and removed | −0.003, removed anyway |
| Windows from ≤15 stars, identity lost | `create_training_data.py:21` | track dropped |
| TESS `KP` mapped to Candidate | `preprocess_merge.py:149` | corrected; `APC`/`FA` no longer dropped |
| Batch-median imputation inside feature building | `build_features` | train/serve skew; **TESS transfer 0.7619 → 0.8376** |
