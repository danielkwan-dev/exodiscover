import numpy as np
import pandas as pd
import pytest

from exodiscover.data.splits import grouped_train_test_split
from exodiscover.experiments import ablation, overfit, transfer
from exodiscover.features.tabular import FEATURE_COLUMNS


def test_collapse_maps_confirmed_and_candidate_to_planetlike():
    out = ablation.collapse_to_planetlike(np.array([0, 1, 2, 0, 2]))
    assert out.tolist() == [0, 1, 1, 0, 1]


def test_leakage_ablation_covers_all_three_setups(koi_sample):
    rows = ablation.run_leakage_ablation(koi_sample)
    assert {r["setup"] for r in rows} == {
        "leaky_random_split",
        "clean_random_split",
        "clean_grouped_split",
    }
    for row in rows:
        assert 0.0 <= row["roc_auc"] <= 1.0
        assert row["note"]


def test_leaky_setup_scores_higher_than_the_honest_one(koi_sample):
    """If this fails, the firewall is misconfigured: adding the Robovetter's
    own verdict columns must make the score go up."""
    by_setup = {r["setup"]: r for r in ablation.run_leakage_ablation(koi_sample)}
    assert by_setup["leaky_random_split"]["roc_auc"] > by_setup["clean_grouped_split"]["roc_auc"]


def test_framing_ablation_reports_all_three_framings(koi_sample):
    rows = ablation.run_framing_ablation(koi_sample)
    assert {r["framing"] for r in rows} == {"3class", "binary", "diagnostic"}
    for row in rows:
        assert 0.0 <= row["pr_auc"] <= 1.0
        assert row["target"]


def test_ablation_attaches_multiplicity_to_a_raw_frame(koi_sample):
    """The ablations are handed the catalog directly, not a prepared frame.

    build_features deliberately will not count KOIs per star itself, because
    counting inside a frame makes a row's features depend on its batch. If the
    caller has not attached the count, n_kois_on_star comes out entirely NaN --
    and HistGradientBoosting cannot bin a column with no distinct values. It
    fails with "window shape cannot be larger than input array shape" from
    inside numpy, which names neither the column nor the cause.
    """
    X = ablation.prepare_features(ablation.with_multiplicity(koi_sample))
    assert X["n_kois_on_star"].notna().all()


def test_prepare_features_rejects_a_wholly_missing_column(koi_sample):
    """An empty feature column is a caller error, and should say so."""
    without_radius = koi_sample.drop(columns=["koi_prad"])
    with pytest.raises(ValueError, match="log_prad"):
        ablation.prepare_features(ablation.with_multiplicity(without_radius))


def test_generalisation_gap_reports_both_sides(koi_binary):
    """The overfitting question is train-minus-test, not the size of test."""
    X, y, groups = koi_binary
    X_tr, X_te, y_tr, y_te = grouped_train_test_split(X, y, groups)
    result = overfit.generalisation_gap(overfit.default_model(), X_tr, y_tr, X_te, y_te)

    assert result["train_roc_auc"] >= result["test_roc_auc"]
    assert result["gap"] == pytest.approx(
        result["train_roc_auc"] - result["test_roc_auc"]
    )
    assert result["overfit"] is (result["gap"] > overfit.OVERFIT_GAP)


def test_learning_curve_is_ordered_and_grows_with_data(koi_binary):
    X, y, groups = koi_binary
    X_tr, X_te, y_tr, y_te = grouped_train_test_split(X, y, groups)
    rows = overfit.learning_curve(
        overfit.default_model(), X_tr, y_tr, groups.loc[X_tr.index], X_te, y_te
    )

    fractions = [r["fraction"] for r in rows]
    assert fractions == sorted(fractions)
    assert rows[-1]["n_train_rows"] > rows[0]["n_train_rows"]
    assert all(0.0 <= r["roc_auc"] <= 1.0 for r in rows)


def test_learning_curve_holds_out_the_same_rows_throughout(koi_binary):
    """Every point must be measured against one fixed held-out set.

    A curve whose test set moves between points measures two things at once
    and cannot show whether more training data helped.
    """
    X, y, groups = koi_binary
    X_tr, X_te, y_tr, y_te = grouped_train_test_split(X, y, groups)
    rows = overfit.learning_curve(
        overfit.default_model(), X_tr, y_tr, groups.loc[X_tr.index], X_te, y_te
    )
    assert {r["n_test_rows"] for r in rows} == {len(y_te)}


def test_shared_features_are_a_subset_of_the_kepler_features():
    assert set(transfer.SHARED_FEATURES) <= set(FEATURE_COLUMNS)
    assert len(transfer.SHARED_FEATURES) >= 8


def test_ambiguous_toi_dispositions_are_excluded():
    toi = pd.DataFrame(
        {
            "tid": [1, 2, 3, 4, 5, 6],
            "tfopwg_disp": ["CP", "KP", "FP", "FA", "PC", "APC"],
            "pl_orbper": [3.0] * 6,
            "pl_trandep": [500.0] * 6,
            "pl_trandurh": [2.0] * 6,
            "pl_rade": [2.0] * 6,
            "st_teff": [5500.0] * 6,
            "st_logg": [4.4] * 6,
            "st_rad": [1.0] * 6,
            "pl_insol": [10.0] * 6,
            "pl_eqt": [600.0] * 6,
            "st_tmag": [11.0] * 6,
        }
    )
    X, y = transfer.toi_to_common(toi)
    assert len(X) == 4, "PC and APC are unvetted and must not become labels"
    assert y.tolist() == [1, 1, 0, 0], "KP is a confirmed planet, not a candidate"


def test_transfer_reports_both_domains(koi_sample, toi_sample):
    result = transfer.run_transfer(koi_sample, toi_sample).metrics
    assert 0.0 <= result["zero_shot"]["roc_auc"] <= 1.0
    assert 0.0 <= result["in_domain"]["roc_auc"] <= 1.0
    assert result["zero_shot"]["n"] > 0
    assert set(result["in_domain"]) >= {"n", "base_rate", "roc_auc", "pr_auc", "brier"}


def test_transfer_reports_accuracy_against_its_own_baseline(koi_sample, toi_sample):
    """The headline figure is an accuracy, so it has to ship with its baseline.

    Kepler is 63% false positives and TESS is nearly balanced, so the two
    accuracies are not comparable on their own: guessing the majority class
    already scores 63% on one and 51% on the other. Reporting the baseline
    beside each is what stops the comparison being misread.
    """
    result = transfer.run_transfer(koi_sample, toi_sample).metrics
    for domain in ("in_domain", "zero_shot"):
        block = result[domain]
        assert 0.0 <= block["accuracy"] <= 1.0
        assert 0.0 <= block["majority_baseline"] <= 1.0
        assert block["majority_baseline"] == pytest.approx(
            max(block["base_rate"], 1 - block["base_rate"])
        )
