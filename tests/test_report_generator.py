"""
Unit tests for AnnotatorIQ ReportGenerator (Markdown & JSON Audit Reports).
"""

from annotatoriq.schema import (
    QualityGateResult,
    DefectRecord,
    DefectType,
)
from annotatoriq.report_generator import ReportGenerator


def test_generate_markdown_report_passed():
    gate_res = QualityGateResult(
        passed=True,
        krippendorff_alpha=0.88,
        fleiss_kappa=0.84,
        gold_accuracy=0.92,
        unresolved_dispute_count=0,
        unresolved_dispute_pct=0.0,
        defect_count=0,
        gate_thresholds={
            "min_alpha": 0.70,
            "min_kappa": 0.65,
            "min_gold_accuracy": 0.85,
            "max_unresolved_disputes": 0,
            "max_critical_defects": 0,
            "max_drift_defects": 2,
        },
        failure_reasons=[],
    )
    annotator_stats = {
        "annotator_01": {
            "gold_items_evaluated": 10,
            "gold_accuracy_pct": 95.0,
            "avg_confidence": 0.94,
            "avg_duration_sec": 14.5,
            "reliability_tier": "Tier 1 (Expert >= 90%)",
        }
    }
    defects = []
    fatigue_profiles = {
        "annotator_01": {
            "fatigue_score": 0.12,
            "speed_drop_ratio": 0.05,
            "status": "NORMAL",
        }
    }

    report_md = ReportGenerator.generate_markdown_report(
        gate_res,
        annotator_stats,
        defects,
        fatigue_profiles=fatigue_profiles,
        dataset_name="batch_v2_clean",
    )

    assert "PASSED ALL QUALITY GATES" in report_md
    assert "batch_v2_clean" in report_md
    assert "Krippendorff's Alpha" in report_md
    assert "Worker Fatigue & Entropy Drift Telemetry" in report_md
    assert "annotator_01" in report_md
    assert "NORMAL" in report_md


def test_generate_markdown_report_failed():
    gate_res = QualityGateResult(
        passed=False,
        krippendorff_alpha=0.55,
        fleiss_kappa=0.48,
        gold_accuracy=0.72,
        unresolved_dispute_count=3,
        unresolved_dispute_pct=15.0,
        defect_count=5,
        gate_thresholds={
            "min_alpha": 0.70,
            "min_kappa": 0.65,
            "min_gold_accuracy": 0.85,
            "max_unresolved_disputes": 0,
            "max_critical_defects": 0,
            "max_drift_defects": 2,
        },
        failure_reasons=[
            "Krippendorff's Alpha (0.550) is below minimum threshold (0.70).",
            "Average Gold Standard Accuracy (72.0%) is below minimum (85.0%).",
        ],
    )
    annotator_stats = {}
    defects = [
        DefectRecord(
            item_id="item_99",
            annotator_id="a_bad",
            defect_type=DefectType.SPEED_ANOMALY,
            description="Annotated in under 0.5s",
            severity="CRITICAL",
        )
    ]

    report_md = ReportGenerator.generate_markdown_report(
        gate_res,
        annotator_stats,
        defects,
        dataset_name="batch_v2_anomalous",
    )

    assert "FAILED QUALITY GATES" in report_md
    assert "Active Release Blockers" in report_md
    assert "Krippendorff's Alpha (0.550) is below minimum threshold" in report_md
    assert "item_99" in report_md


def test_generate_json_report():
    gate_res = QualityGateResult(
        passed=True,
        krippendorff_alpha=0.88,
        fleiss_kappa=0.84,
        gold_accuracy=0.92,
        unresolved_dispute_count=0,
        unresolved_dispute_pct=0.0,
        defect_count=0,
        gate_thresholds={"min_alpha": 0.70},
        failure_reasons=[],
    )
    json_report = ReportGenerator.generate_json_report(
        gate_res,
        annotator_stats={"a1": {"gold_accuracy_pct": 100.0}},
        defects=[],
        fatigue_profiles={"a1": {"fatigue_score": 0.05, "status": "NORMAL"}},
        dataset_name="release_candidate_01",
    )

    assert json_report["dataset_name"] == "release_candidate_01"
    assert json_report["gate_decision"] == "PASSED"
    assert json_report["passed"] is True
    assert json_report["metrics"]["krippendorff_alpha"] == 0.88
    assert "a1" in json_report["fatigue_telemetry"]
