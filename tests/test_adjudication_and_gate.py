"""
Unit tests for Adjudication and QualityGate release gating.
"""

from annotatoriq.schema import (
    AnnotationItem,
    TaskType,
)
from annotatoriq.adjudication import AdjudicationEngine
from annotatoriq.release_gate import QualityGate


def test_dispute_identification():
    items = [
        AnnotationItem(item_id="item_1", annotator_id="a1", task_type=TaskType.CLASSIFICATION, label="positive"),
        AnnotationItem(item_id="item_1", annotator_id="a2", task_type=TaskType.CLASSIFICATION, label="negative"),
        AnnotationItem(item_id="item_2", annotator_id="a1", task_type=TaskType.CLASSIFICATION, label="neutral"),
        AnnotationItem(item_id="item_2", annotator_id="a2", task_type=TaskType.CLASSIFICATION, label="neutral"),
    ]
    disputes = AdjudicationEngine.identify_disputes(items)
    assert len(disputes) == 1
    assert disputes[0]["item_id"] == "item_1"


def test_adjudication_export():
    items = [
        AnnotationItem(item_id="item_1", annotator_id="a1", task_type=TaskType.CLASSIFICATION, label="positive", input_text="Sample"),
    ]
    adj = AdjudicationEngine.resolve_dispute(
        item_id="item_1",
        task_type=TaskType.CLASSIFICATION,
        disputed_options=[],
        final_decision={"label": "neutral"},
        adjudicator_id="sme_lead",
        rationale="Context shows neutral intent.",
    )
    release = AdjudicationEngine.export_adjudicated_dataset(items, [adj])
    assert len(release) == 1
    assert release[0]["final_label"] == "neutral"
    assert release[0]["provenance"] == "ADJUDICATED_EXPERT"


def test_quality_gate_passes_on_unanimous():
    items = [
        AnnotationItem(item_id="i1", annotator_id="a1", task_type=TaskType.CLASSIFICATION, label="positive", duration_seconds=10.0),
        AnnotationItem(item_id="i1", annotator_id="a2", task_type=TaskType.CLASSIFICATION, label="positive", duration_seconds=10.0),
        AnnotationItem(item_id="i2", annotator_id="a1", task_type=TaskType.CLASSIFICATION, label="negative", duration_seconds=10.0),
        AnnotationItem(item_id="i2", annotator_id="a2", task_type=TaskType.CLASSIFICATION, label="negative", duration_seconds=10.0),
    ]
    gate = QualityGate()
    res = gate.evaluate_dataset(items)
    assert res.passed is True
    assert res.unresolved_dispute_count == 0


def test_confidence_weighted_consensus():
    items = [
        AnnotationItem(item_id="i1", annotator_id="a1", task_type=TaskType.CLASSIFICATION, label="positive", confidence=0.95),
        AnnotationItem(item_id="i1", annotator_id="a2", task_type=TaskType.CLASSIFICATION, label="positive", confidence=0.90),
        AnnotationItem(item_id="i1", annotator_id="a3", task_type=TaskType.CLASSIFICATION, label="negative", confidence=0.20),
        AnnotationItem(item_id="i2", annotator_id="a1", task_type=TaskType.CLASSIFICATION, label="positive", confidence=0.52),
        AnnotationItem(item_id="i2", annotator_id="a2", task_type=TaskType.CLASSIFICATION, label="negative", confidence=0.48),
    ]
    consensus = AdjudicationEngine.compute_confidence_weighted_consensus(items)
    assert consensus["i1"]["consensus_label"] == "positive"
    assert consensus["i1"]["requires_escalation"] is False
    assert consensus["i1"]["margin_of_victory"] > 0.70
    assert consensus["i2"]["requires_escalation"] is True


def test_audit_adjudicator_overturn_rates():
    items = [
        AnnotationItem(item_id="i1", annotator_id="good_worker", task_type=TaskType.CLASSIFICATION, label="positive"),
        AnnotationItem(item_id="i1", annotator_id="noisy_worker", task_type=TaskType.CLASSIFICATION, label="negative"),
        AnnotationItem(item_id="i2", annotator_id="good_worker", task_type=TaskType.CLASSIFICATION, label="neutral"),
        AnnotationItem(item_id="i2", annotator_id="noisy_worker", task_type=TaskType.CLASSIFICATION, label="positive"),
    ]
    adj1 = AdjudicationEngine.resolve_dispute(
        item_id="i1",
        task_type=TaskType.CLASSIFICATION,
        disputed_options=[],
        final_decision={"label": "positive"},
        adjudicator_id="expert_1",
        rationale="Clear positive sentiment",
    )
    adj2 = AdjudicationEngine.resolve_dispute(
        item_id="i2",
        task_type=TaskType.CLASSIFICATION,
        disputed_options=[],
        final_decision={"label": "neutral"},
        adjudicator_id="expert_1",
        rationale="Factual statement",
    )
    audit = AdjudicationEngine.audit_adjudicator_overturn_rates(items, [adj1, adj2])
    assert audit["good_worker"]["overturn_rate"] == 0.0
    assert audit["good_worker"]["calibration_flag"] == "ALIGNED"
    assert audit["noisy_worker"]["overturn_rate"] == 1.0
    assert audit["noisy_worker"]["calibration_flag"] == "RECALIBRATE"

