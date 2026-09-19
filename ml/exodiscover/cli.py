"""Command line entrypoint.

``exo train`` is the one command that matters: it runs the ladder, tunes the
winner, calibrates it, evaluates on held-out stars, runs both ablations and
the transfer study, and writes a single metrics.json that the API and the web
dashboard both read.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import typer

from exodiscover import skymap
from exodiscover.config import settings
from exodiscover.data import ingest
from exodiscover.data.schema import assert_no_derived_leakage
from exodiscover.data.splits import grouped_train_test_split
from exodiscover.discover import rank_candidates
from exodiscover.evaluate import (
    bootstrap_ci,
    evaluate_binary,
    plot_all,
    reliability_curve,
    write_metrics,
)
from exodiscover.experiments import ablation, overfit, transfer
from exodiscover.explain import global_importance, shap_matrix
from exodiscover.features.tabular import FEATURE_COLUMNS, add_multiplicity, build_features
from exodiscover.models import train as train_mod
from exodiscover.models.registry import TUNABLE

app = typer.Typer(
    help="ExoDiscover - exoplanet classification over NASA catalogs.",
    add_completion=False,
)

MODEL_VERSION = "0.1.0"

#: If the leakage-free pipeline scores within this margin of the deliberately
#: leaky one, something that encodes the label is still reaching the model.
LEAKAGE_ALARM_GAP = 0.005


def _read(name: str) -> pd.DataFrame:
    df = pd.read_csv(settings.raw_dir / f"{name}.csv", comment="#", low_memory=False)
    # Multiplicity must be counted over the whole catalog, before any
    # label-based filtering, or training and serving disagree on what it means.
    return add_multiplicity(df)


@app.command("ingest")
def ingest_cmd(force: bool = typer.Option(False, help="Re-download even if cached.")) -> None:
    """Download the Kepler, TESS, and K2 catalogs from the NASA archive."""
    for table in ingest.TABLES:
        df = ingest.fetch(table, force=force)
        typer.echo(f"{table}: {len(df)} rows")


@app.command("train")
def train_cmd(
    fast: bool = typer.Option(False, help="Skip Optuna tuning; for CI smoke runs."),
    trials: int = typer.Option(40, help="Optuna trials for the winning family."),
) -> None:
    """Train, evaluate, and write every artifact."""
    koi = _read("koi")
    resolved = koi["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])
    binary = koi[resolved].reset_index(drop=True)
    X = build_features(binary)
    y = (binary["koi_disposition"] == "CONFIRMED").astype(int)
    groups = binary["kepid"]

    # build_features drops the Robovetter columns by name, which cannot see one
    # that arrived under a different name. Audit the values once, here, against
    # the raw frame that still carries them.
    assert_no_derived_leakage(X, binary)

    # The split comes first, and everything that *chooses* anything -- the
    # ladder, the Optuna search -- sees only the training stars. Picking a
    # model on rows that later produce the reported score is the same class of
    # error as a leaky feature: the number comes back optimistic and nothing in
    # the run says so. The held-out stars are touched once, at evaluation.
    X_tr, X_te, y_tr, y_te = grouped_train_test_split(X, y, groups)
    train_groups = groups.loc[X_tr.index]
    test_groups = groups.loc[X_te.index]
    typer.echo(
        f"training on {len(X_tr)} KOIs across {train_groups.nunique()} stars; "
        f"{len(X_te)} KOIs across {test_groups.nunique()} stars held out"
    )

    ladder = train_mod.fit_candidates(X_tr, y_tr, train_groups)
    best = ladder[0]
    typer.echo(
        f"best on the ladder: {best.name} "
        f"(grouped CV PR-AUC {best.cv_scores['pr_auc']:.4f})"
    )

    estimator = best.estimator
    selected = best.name
    if not fast:
        # Tune the best *tunable* family even when an ensemble tops the ladder,
        # then keep whichever actually scores higher. Skipping the search just
        # because an untuned ensemble edged ahead would leave the comparison
        # unfair -- the ensemble's members would never have been tuned at all.
        tunable = next((r for r in ladder if r.name in TUNABLE), None)
        if tunable is not None:
            tuned = train_mod.tune_best(
                X_tr, y_tr, train_groups, tunable.name, n_trials=trials
            )
            tuned_scores = train_mod.score_estimator(tuned, X_tr, y_tr, train_groups)
            typer.echo(
                f"tuned {tunable.name}: PR-AUC {tunable.cv_scores['pr_auc']:.4f} "
                f"-> {tuned_scores['pr_auc']:.4f} ({trials} trials)"
            )
            if tuned_scores["pr_auc"] > best.cv_scores["pr_auc"]:
                estimator, selected = tuned, f"{tunable.name} (tuned)"
    typer.echo(f"selected: {selected}")

    model, calibration = train_mod.build_calibrated(estimator, X_tr, y_tr, train_groups)
    typer.echo(
        f"calibration: {calibration['chosen']} "
        f"(Brier isotonic {calibration['brier_by_method']['isotonic']:.4f} vs "
        f"sigmoid {calibration['brier_by_method']['sigmoid']:.4f})"
    )

    # Overfitting is a gap, not a level. Measure it on the uncalibrated
    # estimator: the calibration wrapper is fitted on a slice of the training
    # data, so scoring it against that same data would understate the gap.
    gap = overfit.generalisation_gap(estimator, X_tr, y_tr, X_te, y_te)
    curve = overfit.learning_curve(estimator, X_tr, y_tr, train_groups, X_te, y_te)
    verdict = "OVERFIT" if gap["overfit"] else "generalises"
    typer.echo(
        f"generalisation: train {gap['train_roc_auc']:.4f} vs "
        f"held-out {gap['test_roc_auc']:.4f} "
        f"(gap {gap['gap']:.4f}, threshold {gap['threshold']:.2f}) -> {verdict}"
    )

    test_metrics = evaluate_binary(model, X_te, y_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    test_metrics["ci95"] = bootstrap_ci(y_te, y_prob, test_groups)
    roc_lo, roc_hi = test_metrics["ci95"]["roc_auc"]
    typer.echo(
        f"held-out stars: ROC-AUC {test_metrics['roc_auc']:.4f} "
        f"[{roc_lo:.4f}, {roc_hi:.4f}]  "
        f"PR-AUC {test_metrics['pr_auc']:.4f} Brier {test_metrics['brier']:.4f}"
    )
    # Leakage guard. A fixed accuracy threshold is the wrong alarm here: this
    # task is legitimately easy, because the false-positive population is
    # dominated by eclipsing binaries whose implied planet radius is physically
    # impossible. What would actually indicate leakage is the clean pipeline
    # closing the gap on the deliberately-leaky one, so that is what we check.
    leakage_rows = ablation.run_leakage_ablation(koi)
    by_setup = {r["setup"]: r for r in leakage_rows}
    leaky = by_setup["leaky_random_split"]["roc_auc"]
    clean = by_setup["clean_grouped_split"]["roc_auc"]
    if leaky - clean < LEAKAGE_ALARM_GAP:
        typer.secho(
            f"WARNING: the clean pipeline ({clean:.4f}) is within {LEAKAGE_ALARM_GAP} of the "
            f"deliberately-leaky one ({leaky:.4f}). A leaky feature may have survived; "
            "check docs/LEAKAGE.md before trusting this run.",
            fg=typer.colors.RED,
        )
    else:
        typer.echo(f"leakage guard ok: clean {clean:.4f} vs leaky {leaky:.4f}")

    toi_path = settings.raw_dir / "toi.csv"
    transfer_result = transfer.run_transfer(koi, _read("toi")) if toi_path.exists() else None
    transfer_block = (
        transfer_result.metrics
        if transfer_result is not None
        else {"in_domain": {}, "zero_shot": {}}
    )

    payload = {
        "model": {
            "name": selected,
            "version": MODEL_VERSION,
            "trained_at": datetime.now(UTC).isoformat(),
            "framing": "binary",
            "tuned": not fast,
            "test_size": settings.test_size,
            "n_train_rows": int(len(X_tr)),
            "n_train_stars": int(train_groups.nunique()),
            "n_test_rows": int(len(X_te)),
            "n_test_stars": int(test_groups.nunique()),
        },
        # Answers "is 0.98 too good to be true" with measurements rather than
        # assertion: the train-minus-held-out gap, and whether the score
        # survives training on a fraction of the stars.
        "overfitting": {"gap": gap, "learning_curve": curve},
        "features": FEATURE_COLUMNS,
        "test": test_metrics,
        "cv": best.cv_scores,
        "calibration": calibration,
        "ladder": [{"name": r.name, **r.cv_scores} for r in ladder],
        "ablation": {
            "leakage": leakage_rows,
            "framing": ablation.run_framing_ablation(koi),
        },
        "transfer": transfer_block,
        "importance": global_importance(model, X_te),
        "reliability": reliability_curve(y_te.to_numpy(), y_prob),
    }
    metrics_path = write_metrics(payload, settings.metrics_path)
    plot_all(y_te, y_prob, settings.metrics_dir)
    typer.echo(f"wrote {metrics_path}")

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = settings.models_dir / "model.joblib"
    joblib.dump(
        {"model": model, "features": FEATURE_COLUMNS, "version": MODEL_VERSION},
        bundle_path,
        compress=3,
    )
    size_mb = bundle_path.stat().st_size / 1024 / 1024
    typer.echo(f"wrote {bundle_path} ({size_mb:.1f} MB)")

    # The shared-feature model, kept because the reported zero-shot figure is
    # its number and the sky map has to show the same model the panel quotes.
    # Two artifacts with two stated jobs: this one is the cross-mission
    # scorer, model.joblib is the tuned Kepler triage model.
    if transfer_result is not None:
        shared_path = settings.models_dir / "transfer.joblib"
        joblib.dump(
            {
                "model": transfer_result.model,
                "features": transfer.SHARED_FEATURES,
                "version": MODEL_VERSION,
            },
            shared_path,
            compress=3,
        )
        zero_shot = transfer_block["zero_shot"]
        typer.echo(
            f"wrote {shared_path} (zero-shot on TESS: "
            f"{zero_shot['accuracy']:.4f} accuracy vs "
            f"{zero_shot['majority_baseline']:.4f} baseline)"
        )

    ranked = rank_candidates(model, koi, top_n=50)
    ranked_path = settings.metrics_dir / "top_candidates.csv"
    ranked.to_csv(ranked_path, index=False)
    typer.echo(f"wrote {ranked_path} ({len(ranked)} candidates)")


#: How many SHAP terms the detail panel shows for a selected planet.
REASONS_SHOWN = 3


@app.command("skymap")
def skymap_cmd() -> None:
    """Place both catalogs in space and write the sky map the web view reads."""
    koi = ingest.load_cached("koi")
    stellar = ingest.load_cached("stellar")
    toi = ingest.load_cached("toi")

    # One model for both missions, and specifically the model whose zero-shot
    # figure the app reports. The shipped 17-feature model cannot score TESS --
    # only 11 features exist in both catalogs -- and scoring the two missions
    # with different models would make their probabilities incomparable, which
    # is the one thing this view puts side by side. Loading the persisted
    # artifact rather than refitting one here is what keeps the number on
    # screen and the number in metrics.json describing the same estimator.
    shared_path = settings.models_dir / "transfer.joblib"
    if not shared_path.exists():
        raise typer.BadParameter(
            f"no shared-feature model at {shared_path}. Run `exo train` first: "
            "the sky map scores both missions with the model whose zero-shot "
            "accuracy the app reports, rather than fitting its own."
        )
    bundle = joblib.load(shared_path)
    model, shared = bundle["model"], list(bundle["features"])

    # Every Kepler object with a resolved disposition was in this model's
    # training data. Those rows get an out-of-fold probability instead, so
    # nothing on screen is a model's opinion of a row it already saw. Kepler
    # candidates and every TESS object were never in training and are scored
    # directly.
    counted = add_multiplicity(koi)
    resolved = counted[
        counted["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])
    ].reset_index(drop=True)
    oof = skymap.out_of_fold_probabilities(
        model,
        build_features(resolved)[shared],
        (resolved["koi_disposition"] == "CONFIRMED").astype(int),
        resolved["kepid"],
    )
    out_of_fold = dict(zip(resolved["kepoi_name"], oof, strict=True))
    typer.echo(
        f"scoring both missions with the {len(shared)}-feature shared model; "
        f"{len(out_of_fold)} trained-on Kepler rows scored out of fold"
    )

    def score(X: pd.DataFrame, frame: pd.DataFrame):
        direct = model.predict_proba(X[shared])[:, 1]
        held = frame["name"].map(out_of_fold)
        return np.where(held.notna(), held.to_numpy(dtype=float), direct)

    def reasons(X: pd.DataFrame) -> list[str]:
        """Top contributions per row, precomputed so a click costs nothing."""
        values = shap_matrix(model, X[shared])
        order = np.argsort(-np.abs(values), axis=1)[:, :REASONS_SHOWN]
        return [
            ";".join(f"{shared[j]}:{values[i, j]:+.2f}" for j in row)
            for i, row in enumerate(order)
        ]

    table = skymap.build_skymap(koi, stellar, toi, score, reasons)

    settings.metrics_dir.mkdir(parents=True, exist_ok=True)
    path = settings.metrics_dir / "sky_map.csv"
    table.to_csv(path, index=False)

    ly = table["dist_pc"] * skymap.LY_PER_PARSEC
    by_mission = table["mission"].value_counts().to_dict()
    typer.echo(f"wrote {path} ({len(table)} objects: {by_mission})")
    typer.echo(
        f"  nearest {ly.min():,.0f} ly, median {ly.median():,.0f} ly, "
        f"farthest {ly.max():,.0f} ly"
    )


@app.command("eval")
def eval_cmd() -> None:
    """Print the stored held-out metrics."""
    payload = json.loads(settings.metrics_path.read_text())
    typer.echo(json.dumps({"model": payload["model"], "test": payload["test"]}, indent=2))


@app.command("predict")
def predict_cmd(csv: Path = typer.Argument(..., help="CSV of KOI rows to score.")) -> None:
    """Score a CSV of KOI rows with the production model."""
    bundle = joblib.load(settings.models_dir / "model.joblib")
    X = build_features(pd.read_csv(csv, comment="#", low_memory=False))[bundle["features"]]
    for i, p in enumerate(bundle["model"].predict_proba(X)[:, 1]):
        typer.echo(f"row {i}: planet probability {p:.4f}")


if __name__ == "__main__":
    app()
