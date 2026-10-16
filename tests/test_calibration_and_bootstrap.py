import pytest
from annotatoriq.metrics import AgreementMetrics


def test_bootstrap_agreement_ci_perfect():
    r1 = ["POS", "NEG", "NEU", "POS", "NEG"] * 5
    r2 = ["POS", "NEG", "NEU", "POS", "NEG"] * 5

    res = AgreementMetrics.bootstrap_agreement_ci(r1, r2, n_bootstraps=50, ci=0.95, seed=123)
    assert res["point_estimate"] == 1.0
    assert res["ci_lower"] == 1.0
    assert res["ci_upper"] == 1.0


def test_bootstrap_agreement_ci_realistic():
    r1 = ["POS", "POS", "NEG", "NEU", "POS", "NEG", "NEU", "POS"] * 4
    r2 = ["POS", "NEG", "NEG", "NEU", "POS", "NEG", "POS", "POS"] * 4

    res = AgreementMetrics.bootstrap_agreement_ci(r1, r2, n_bootstraps=100, ci=0.95, seed=42)
    assert 0.0 < res["point_estimate"] < 1.0
    assert res["ci_lower"] <= res["point_estimate"] <= res["ci_upper"]
    assert res["std_error"] >= 0.0


def test_compute_brier_and_calibration_perfect():
    probs = [1.0, 0.0, 1.0, 0.0]
    outcomes = [1, 0, 1, 0]

    cal = AgreementMetrics.compute_brier_and_calibration(probs, outcomes)
    assert cal["brier_score"] == 0.0
    assert cal["mean_calibration_error"] == 0.0


def test_compute_brier_and_calibration_uncalibrated():
    probs = [0.9, 0.8, 0.7, 0.8]
    outcomes = [0, 0, 0, 0]

    cal = AgreementMetrics.compute_brier_and_calibration(probs, outcomes)
    assert cal["brier_score"] > 0.4
    assert cal["mean_calibration_error"] > 0.5
