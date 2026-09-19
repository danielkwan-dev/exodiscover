"""The sky-map artifact: both missions, real positions, nothing invented."""

import numpy as np
import pandas as pd
import pytest

from exodiscover import skymap


def _scorer(p: float = 0.5):
    return lambda X, frame: np.full(len(X), p)


def _reasoner(X: pd.DataFrame) -> list[str]:
    return ["log_prad:+1.00;duration_ratio:-0.50"] * len(X)


@pytest.fixture
def stellar(koi_sample: pd.DataFrame) -> pd.DataFrame:
    """A stand-in for Q1_Q17_DR25_KS: one distance per star, some unusable."""
    kepids = koi_sample["kepid"].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(0)
    dist = pd.Series(rng.uniform(250.0, 1900.0, size=len(kepids)))
    # The real table contains both of these, and neither is a distance.
    dist.iloc[0] = 0.0
    dist.iloc[1] = np.nan
    return pd.DataFrame({"kepid": kepids, "dist": dist, "dist_err1": dist * 0.19})


@pytest.fixture
def built(koi_sample, stellar, toi_sample):
    return skymap.build_skymap(koi_sample, stellar, toi_sample, _scorer(), _reasoner)


def test_columns_are_exactly_the_contract(built):
    assert list(built.columns) == skymap.SKYMAP_COLUMNS


def test_both_missions_are_present(built):
    """TESS is what makes the view surround you: Kepler stared at one 22x16
    degree patch, TESS covered the whole sky."""
    assert set(built["mission"]) == {"Kepler", "TESS"}
    assert (built["mission"] == "Kepler").sum() > 0
    assert (built["mission"] == "TESS").sum() > 0


def test_every_row_can_actually_be_placed(built):
    for column in ("ra", "dec", "dist_pc"):
        assert built[column].notna().all()
    assert (built["dist_pc"] > 0).all()


def test_tess_covers_sky_kepler_does_not(built):
    """A sanity check on the thing the map exists to show."""
    tess = built[built["mission"] == "TESS"]
    kepler = built[built["mission"] == "Kepler"]
    assert tess["dec"].min() < kepler["dec"].min()
    assert tess["ra"].max() - tess["ra"].min() > kepler["ra"].max() - kepler["ra"].min()


def test_tess_dispositions_are_mapped_to_the_shared_vocabulary(built):
    """The two archives disagree on names. CP and KP are both confirmed
    planets, PC and APC are unvetted, FP and FA are neither."""
    assert set(built["disposition"]) <= {"CONFIRMED", "CANDIDATE", "FALSE POSITIVE"}
    tess = built[built["mission"] == "TESS"]
    assert len(set(tess["disposition"])) > 1


def test_unusable_distances_are_dropped_not_repaired(koi_sample, stellar, toi_sample):
    out = skymap.build_skymap(koi_sample, stellar, toi_sample, _scorer(), _reasoner)
    bad = set(stellar.loc[stellar["dist"].isna() | (stellar["dist"] <= 0), "kepid"])
    kepler_ids = set(out.loc[out["mission"] == "Kepler", "star_id"])
    assert not (kepler_ids & bad)


def test_probability_comes_from_one_model_for_both_missions(koi_sample, stellar, toi_sample):
    """Scoring the two catalogs with different models would make the numbers
    incomparable; only the 11 shared features can serve both."""
    out = skymap.build_skymap(koi_sample, stellar, toi_sample, _scorer(0.25), _reasoner)
    assert (out["probability"] == 0.25).all()


def test_reasons_are_precomputed_and_parseable(built):
    """The panel reads these directly, so no scoring round-trip on click."""
    assert built["top_reasons"].str.len().gt(0).all()
    first = built["top_reasons"].iloc[0]
    for part in first.split(";"):
        name, value = part.rsplit(":", 1)
        assert name
        float(value)


def test_out_of_fold_probabilities_do_not_memorise():
    """A model scored on rows it trained on can memorise noise; scored out of
    fold it cannot. Random labels over random features make the difference
    unambiguous: in-sample separation is perfect, out-of-fold is chance.

    This is what keeps the sky map honest. Every Kepler object with a resolved
    disposition was in the shared model's training data, so scoring it with
    that model would put an in-sample fit on screen and call it a prediction.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = pd.Series(rng.integers(0, 2, size=n))
    groups = pd.Series(np.arange(n))  # one row per star

    model = DecisionTreeClassifier(random_state=0)
    in_sample = model.fit(X, y).predict_proba(X)[:, 1]
    out_of_fold = skymap.out_of_fold_probabilities(model, X, y, groups)

    assert len(out_of_fold) == n
    assert roc_auc_score(y, in_sample) > 0.99, "the memoriser did not memorise"
    assert roc_auc_score(y, out_of_fold) < 0.65, (
        "out-of-fold scores separate random labels, so a fold saw its own rows"
    )


def test_each_object_is_scored_with_its_own_features(koi_sample, stellar, toi_sample):
    """Regression: every Kepler object was scored with another object's row.

    `_kepler_frame` builds the frame from the whole catalog, then inner-joins
    the stellar distance table. The join drops the stars with no usable
    distance and hands back a fresh RangeIndex, so selecting features by that
    index returns the *first* N feature rows rather than the features of the
    rows that actually survived. Everything after the first dropped star was
    displaying a different planet's probability and a different planet's SHAP
    terms. A constant scorer cannot see this, which is why it survived.
    """

    def score(X: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
        return X["log_prad"].fillna(-99.0).to_numpy()

    out = skymap.build_skymap(koi_sample, stellar, toi_sample, score, _reasoner)
    kepler = out[out["mission"] == "Kepler"]

    expected = np.log10(
        pd.to_numeric(kepler["radius_earth"], errors="coerce").clip(lower=1e-9)
    )
    pair = pd.DataFrame(
        {"expected": expected.to_numpy(), "got": kepler["probability"].to_numpy()}
    ).dropna()

    assert len(pair) > 0
    wrong = int((~np.isclose(pair["expected"], pair["got"], atol=1e-6)).sum())
    assert wrong == 0, f"{wrong} of {len(pair)} Kepler objects scored with the wrong row"
