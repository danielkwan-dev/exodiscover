import json
import shutil

from typer.testing import CliRunner

from exodiscover.cli import app

runner = CliRunner()

REQUIRED_KEYS = (
    "model",
    "features",
    "test",
    "cv",
    "ladder",
    "ablation",
    "transfer",
    "importance",
    "reliability",
)


def test_train_writes_the_full_artifact_contract(tmp_path, monkeypatch):
    monkeypatch.setattr("exodiscover.cli.settings.root", tmp_path)
    monkeypatch.setattr("exodiscover.evaluate.settings.root", tmp_path)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    shutil.copy("tests/fixtures/koi_sample.csv", raw / "koi.csv")
    shutil.copy("tests/fixtures/toi_sample.csv", raw / "toi.csv")

    result = runner.invoke(app, ["train", "--fast"])
    assert result.exit_code == 0, result.output

    payload = json.loads((tmp_path / "docs" / "metrics" / "metrics.json").read_text())
    for key in REQUIRED_KEYS:
        assert key in payload, f"metrics.json missing required key {key}"

    assert (tmp_path / "models" / "production" / "model.joblib").exists()
    assert (tmp_path / "docs" / "metrics" / "top_candidates.csv").exists()
    for figure in ("pr_curve.png", "reliability.png", "confusion_matrix.png"):
        assert (tmp_path / "docs" / "metrics" / figure).exists()


def test_train_records_star_counts_not_just_row_counts(tmp_path, monkeypatch):
    monkeypatch.setattr("exodiscover.cli.settings.root", tmp_path)
    monkeypatch.setattr("exodiscover.evaluate.settings.root", tmp_path)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    shutil.copy("tests/fixtures/koi_sample.csv", raw / "koi.csv")

    assert runner.invoke(app, ["train", "--fast"]).exit_code == 0
    model_block = json.loads((tmp_path / "docs" / "metrics" / "metrics.json").read_text())["model"]
    assert model_block["n_train_stars"] <= model_block["n_train_rows"]
    assert model_block["framing"] == "binary"


def test_predict_command_is_registered():
    assert runner.invoke(app, ["predict", "--help"]).exit_code == 0


def test_help_lists_every_command():
    output = runner.invoke(app, ["--help"]).output
    for command in ("ingest", "train", "eval", "predict"):
        assert command in output


def _fixture_run(tmp_path, monkeypatch, *, toi: bool = False):
    """Point the CLI at a temp root holding only the committed fixtures."""
    monkeypatch.setattr("exodiscover.cli.settings.root", tmp_path)
    monkeypatch.setattr("exodiscover.evaluate.settings.root", tmp_path)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    shutil.copy("tests/fixtures/koi_sample.csv", raw / "koi.csv")
    if toi:
        shutil.copy("tests/fixtures/toi_sample.csv", raw / "toi.csv")


def _spy_on_selection(monkeypatch) -> dict[str, set]:
    """Record the stars model selection saw, and the stars scored at the end.

    `bootstrap_ci` is the reporting call that receives the held-out groups
    directly, so spying there gets the test stars without recomputing a split
    the test would then have to keep in sync with the code.
    """
    from exodiscover import cli
    from exodiscover.models import train as train_mod

    seen: dict[str, set] = {}

    real_fit = train_mod.fit_candidates
    real_tune = train_mod.tune_best
    real_ci = cli.bootstrap_ci

    def spy_fit(X, y, groups, **kwargs):
        seen["ladder"] = set(groups)
        return real_fit(X, y, groups, **kwargs)

    def spy_tune(X, y, groups, name, **kwargs):
        seen["tuning"] = set(groups)
        return real_tune(X, y, groups, name, **kwargs)

    def spy_ci(y_true, y_prob, groups, **kwargs):
        seen["held_out"] = set(groups)
        return real_ci(y_true, y_prob, groups, **kwargs)

    monkeypatch.setattr("exodiscover.cli.train_mod.fit_candidates", spy_fit)
    monkeypatch.setattr("exodiscover.cli.train_mod.tune_best", spy_tune)
    monkeypatch.setattr("exodiscover.cli.bootstrap_ci", spy_ci)
    return seen


def test_the_ladder_never_sees_a_held_out_star(tmp_path, monkeypatch):
    """Choosing the model on rows that later produce the reported score makes
    that score optimistic. docs/LEAKAGE.md claims this does not happen; this
    is the test that makes the claim true rather than aspirational."""
    _fixture_run(tmp_path, monkeypatch)
    seen = _spy_on_selection(monkeypatch)

    assert runner.invoke(app, ["train", "--fast"]).exit_code == 0

    overlap = seen["ladder"] & seen["held_out"]
    assert not overlap, (
        f"{len(overlap)} star(s) used to pick the model are also in the "
        f"held-out set the reported metrics come from"
    )


def test_tuning_never_sees_a_held_out_star(tmp_path, monkeypatch):
    """Same guarantee for the Optuna search, which --fast skips entirely."""
    _fixture_run(tmp_path, monkeypatch)
    seen = _spy_on_selection(monkeypatch)

    assert runner.invoke(app, ["train", "--trials", "1"]).exit_code == 0

    overlap = seen["tuning"] & seen["held_out"]
    assert not overlap, (
        f"{len(overlap)} star(s) the Optuna search scored against are also in "
        f"the held-out set the reported metrics come from"
    )


def test_train_persists_the_shared_feature_transfer_model(tmp_path, monkeypatch):
    """The 77% headline describes the shared-feature model, so that model has
    to survive the run. Refitting an equivalent one later produces numbers no
    reported figure corresponds to."""
    import joblib

    from exodiscover.experiments.transfer import SHARED_FEATURES

    _fixture_run(tmp_path, monkeypatch, toi=True)
    assert runner.invoke(app, ["train", "--fast"]).exit_code == 0

    bundle = joblib.load(tmp_path / "models" / "production" / "transfer.joblib")
    assert bundle["features"] == SHARED_FEATURES
