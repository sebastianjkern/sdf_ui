"""Exercise the dashboard demo, exported metrics, and checkpoint inference."""

import json
import math
import subprocess
import sys

import pytest

from examples import learned_dashboard
from sdf_ui.learning import GaussianPolicy


def test_dashboard_training_and_checkpoint_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(learned_dashboard, "TRAIN", learned_dashboard.TRAIN[:1])
    monkeypatch.setattr(learned_dashboard, "HELD_OUT", learned_dashboard.HELD_OUT[:1])
    output = tmp_path / "train"
    result = learned_dashboard.run(output, steps=2)
    assert result["steps"] == 2
    assert result["training_image_readbacks"] == 0
    assert math.isfinite(result["last_training_loss"])
    assert (output / "comparison.png").is_file()
    assert (output / "checkpoint.json").is_file()
    policy = GaussianPolicy(4, 3, hidden_dim=8)
    policy.load_state_dict(json.loads((output / "policy.json").read_text()))
    name = result["training_cases"][0]
    before = result["before"][name]["card_widths"]
    after = result["after"][name]["card_widths"]
    assert any(abs(a - b) > 1e-4 for a, b in zip(after, before))
    viewport = result["after"][name]["viewport"][0]
    assert sum(after) == pytest.approx(
        viewport - 2 * learned_dashboard.MARGIN - 2 * learned_dashboard.GAP
    )
    scenario = learned_dashboard.TRAIN[0]
    predicted = learned_dashboard.predict(policy, scenario[1], scenario[2])
    assert [predicted[key].width for key in learned_dashboard.NAMES] == pytest.approx(
        after, abs=1e-5
    )
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.learned_dashboard",
            "--policy",
            str(output / "policy.json"),
            "--predict",
            str(scenario[1]),
            *(str(value) for value in scenario[2]),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    exported = json.loads(cli.stdout)
    assert [
        exported["cards"][key]["width"] for key in learned_dashboard.NAMES
    ] == pytest.approx(after, abs=1e-5)
    assert isinstance(exported["violations"], dict)

    inference = learned_dashboard.run(
        tmp_path / "inference",
        steps=0,
        seed=17,
        resume=output / "checkpoint.json",
        scenarios=[(150, 20, 15, 28)],
    )
    held_out_name = result["held_out_cases"][0]
    assert inference["steps"] == 2
    assert inference["after"][held_out_name]["loss"] == pytest.approx(
        result["after"][held_out_name]["loss"], abs=1e-7
    )
    assert inference["after"]["custom_1"]["viewport"] == [150, learned_dashboard.HEIGHT]
    assert (tmp_path / "inference" / "custom_1_after.png").is_file()
    fresh = learned_dashboard.run(
        tmp_path / "fresh", steps=0, initial_policy=output / "policy.json"
    )
    assert fresh["steps"] == 0
    assert fresh["after"][held_out_name]["loss"] == pytest.approx(
        result["after"][held_out_name]["loss"], abs=1e-7
    )


def test_dashboard_rejects_infeasible_custom_content(tmp_path):
    with pytest.raises(ValueError, match="cannot fit"):
        learned_dashboard.run(tmp_path, steps=0, scenarios=[(80, 40, 40, 40)])


def test_dashboard_early_stopping_uses_distinct_validation_case(tmp_path, monkeypatch):
    monkeypatch.setattr(learned_dashboard, "TRAIN", learned_dashboard.TRAIN[:1])
    monkeypatch.setattr(learned_dashboard, "HELD_OUT", learned_dashboard.HELD_OUT[:1])
    report = learned_dashboard.run(
        tmp_path, steps=4, patience=1, validation_interval=1, min_delta=1.0
    )
    assert report["validation_cases"] == ["validation"]
    assert set(report["validation_cases"]).isdisjoint(report["held_out_cases"])
    assert report["training_steps_this_run"] == 1
    assert report["steps"] == 0
