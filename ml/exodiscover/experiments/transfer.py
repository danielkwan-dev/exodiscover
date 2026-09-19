"""Zero-shot Kepler to TESS transfer.

This answers the challenge's actual ask - can the model analyse *new* data -
by training on Kepler and evaluating on TESS objects it has never seen, using
only the features both catalogs can express. PC and APC objects are excluded
because they are unvetted: they carry no ground truth to score against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from exodiscover.config import settings
from exodiscover.data.schema import TOI_LABEL_MAP, TOI_RESOLVED
from exodiscover.data.splits import grouped_train_test_split
from exodiscover.evaluate import binary_scores
from exodiscover.features import tabular

#: Features constructible from both catalogs. Kepler-only quantities such as
#: koi_model_snr and the KOI multiplicity count have no TESS equivalent.
SHARED_FEATURES: list[str] = [
    "log_period",
    "log_depth",
    "log_prad",
    "log_insol",
    "koi_teq",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "rho_star",
    "a_over_rstar",
    "duration_ratio",
]

#: TESS column -> the KOI name build_features expects.
TOI_RENAME: dict[str, str] = {
    "pl_orbper": "koi_period",
    "pl_trandep": "koi_depth",
    "pl_trandurh": "koi_duration",
    "pl_rade": "koi_prad",
    "pl_insol": "koi_insol",
    "pl_eqt": "koi_teq",
    "st_teff": "koi_steff",
    "st_logg": "koi_slogg",
    "st_rad": "koi_srad",
    "st_tmag": "koi_kepmag",
    "tid": "kepid",
}


def toi_to_common(toi: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Map TESS objects onto the shared feature space with binary labels."""
    resolved = toi[toi["tfopwg_disp"].isin(TOI_RESOLVED)].copy()
    # TOI_LABEL_MAP yields 2 for confirmed and 0 for false positive; the
    # transfer task is binary, so 2 becomes 1.
    y = (resolved["tfopwg_disp"].map(TOI_LABEL_MAP) == 2).astype(int)

    df = resolved.rename(columns=TOI_RENAME)
    X = tabular.build_features(df.reset_index(drop=True))[SHARED_FEATURES]
    return X, y.reset_index(drop=True)


def _domain_scores(y_true: pd.Series, y_prob: np.ndarray) -> dict:
    """Metrics for one domain, with the baseline that makes accuracy readable.

    Accuracy alone does not survive being compared across these two domains.
    Kepler's held-out slice is 63% false positives, so predicting the majority
    class scores 0.63 before the model does anything; TESS is close to balanced,
    where the same strategy scores 0.51. Quoting the two accuracies side by side
    without their baselines overstates the drop -- the base-rate-free comparison
    is ROC-AUC, which is why it stays the headline metric.
    """
    y_pred = (y_prob >= 0.5).astype(int)
    base_rate = float(y_true.mean())
    return {
        "n": int(len(y_true)),
        "base_rate": base_rate,
        "majority_baseline": max(base_rate, 1.0 - base_rate),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        **binary_scores(y_true, y_prob),
    }


@dataclass
class TransferResult:
    """The fitted shared-feature model and what it scored.

    The model is returned rather than discarded because the reported 77% is
    *its* number. Refitting an equivalent one elsewhere -- which the sky map
    used to do -- produces probabilities that no published figure describes.
    """

    model: BaseEstimator
    metrics: dict


def run_transfer(koi: pd.DataFrame, toi: pd.DataFrame) -> TransferResult:
    """Train on Kepler, score held-out Kepler and unseen TESS."""
    src = koi[koi["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].reset_index(drop=True)
    y_src = (src["koi_disposition"] == "CONFIRMED").astype(int)
    X_src = tabular.build_features(src)[SHARED_FEATURES]

    X_tr, X_te, y_tr, y_te = grouped_train_test_split(X_src, y_src, src["kepid"])
    model = HistGradientBoostingClassifier(random_state=settings.random_seed).fit(X_tr, y_tr)

    X_tgt, y_tgt = toi_to_common(toi)

    in_domain = _domain_scores(y_te, model.predict_proba(X_te)[:, 1])
    zero_shot = _domain_scores(y_tgt, model.predict_proba(X_tgt)[:, 1])

    metrics = {
        "shared_features": SHARED_FEATURES,
        "in_domain": in_domain,
        "zero_shot": zero_shot,
        "roc_auc_drop": float(in_domain["roc_auc"] - zero_shot["roc_auc"]),
        # Reported alongside the ROC-AUC drop because they disagree, and the
        # disagreement is the point: the ranking degrades, the probabilities
        # collapse. A model that still orders candidates usefully off-domain
        # but can no longer say how likely any of them is.
        "brier_ratio": float(zero_shot["brier"] / in_domain["brier"]),
    }
    return TransferResult(model=model, metrics=metrics)
