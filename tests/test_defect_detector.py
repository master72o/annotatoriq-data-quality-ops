"""
Unit tests for AnnotatorIQ defect detector and gold evaluation.
"""

from annotatoriq.schema import (
    AnnotationItem,
    GoldItem,
    TaskType,
    DefectType,
    SpanEntity,
    BoundingBox,
)
from annotatoriq.defect_detector import DefectDetector
from annotatoriq.gold_eval import GoldEvaluator


def test_speed_anomaly_detected():
    detector = DefectDetector()
    item = AnnotationItem(
        item_id="task_1",
        annotator_id="ann_quick",
        task_type=TaskType.CLASSIFICATION,
        label="positive",
        duration_seconds=0.8,  # Below 2.0s
    )
    defects = detector.scan_annotations([item])
    assert any(d.defect_type == DefectType.SPEED_ANOMALY for d in defects)


def test_out_of_vocab_detected():
    detector = DefectDetector()
    item = AnnotationItem(
        item_id="task_2",
        annotator_id="ann_01",
        task_type=TaskType.CLASSIFICATION,
        label="invalid_custom_sentiment",
        duration_seconds=10.0,
    )
    defects = detector.scan_annotations([item])
    assert any(d.defect_type == DefectType.OUT_OF_VOCAB for d in defects)


def test_gold_evaluation_scoring():
    gold = [
        GoldItem(item_id="g1", task_type=TaskType.CLASSIFICATION, expected_label="toxic"),
        GoldItem(item_id="g2", task_type=TaskType.CLASSIFICATION, expected_label="safe"),
    ]
    annotations = [
        AnnotationItem(item_id="g1", annotator_id="worker_a", task_type=TaskType.CLASSIFICATION, label="toxic"),
        AnnotationItem(item_id="g2", annotator_id="worker_a", task_type=TaskType.CLASSIFICATION, label="toxic"),  # Incorrect
    ]
    res = GoldEvaluator.evaluate_annotators(annotations, gold)
    assert "worker_a" in res
    assert res["worker_a"]["gold_accuracy"] == 0.50
    assert "Tier 4" in res["worker_a"]["reliability_tier"]


def test_audio_segment_defects():
    detector = DefectDetector()
    # Missing segments list
    item_none = AnnotationItem(
        item_id="aud_1",
        annotator_id="worker_b",
        task_type=TaskType.AUDIO_SEGMENT,
        segments=None,
    )
    defects = detector.scan_annotations([item_none])
    assert any(d.defect_type == DefectType.SCHEMA_INVALID for d in defects)


def test_top_level_package_exports():
    import annotatoriq
    assert hasattr(annotatoriq, "DefectDetector")
    assert hasattr(annotatoriq, "GoldEvaluator")
    assert hasattr(annotatoriq, "AgreementMetrics")
    assert hasattr(annotatoriq, "QualityGate")


def test_gold_evaluation_audio():
    from annotatoriq.schema import AudioSegment
    gold = [
        GoldItem(
            item_id="aud_gold_1",
            task_type=TaskType.AUDIO_SEGMENT,
            expected_segments=[
                AudioSegment(start_sec=0.0, end_sec=2.5, transcript="hello world")
            ],
        )
    ]
    annotations = [
        AnnotationItem(
            item_id="aud_gold_1",
            annotator_id="worker_audio",
            task_type=TaskType.AUDIO_SEGMENT,
            segments=[
                AudioSegment(start_sec=0.0, end_sec=2.5, transcript="hello world")
            ],
        )
    ]
    res = GoldEvaluator.evaluate_annotators(annotations, gold)
    assert "worker_audio" in res
    assert res["worker_audio"]["gold_accuracy"] == 1.0


def test_confidence_anomaly_detected():
    detector = DefectDetector()
    item = AnnotationItem(
        item_id="task_conf_low",
        annotator_id="worker_uncertain",
        task_type=TaskType.CLASSIFICATION,
        label="safe",
        duration_seconds=12.0,
        confidence=0.12,  # Below default 0.20 threshold
    )
    defects = detector.scan_annotations([item])
    assert any(d.defect_type == DefectType.CONFIDENCE_ANOMALY for d in defects)


def test_repetitive_pattern_straight_lining_detected():
    detector = DefectDetector()
    # 6 consecutive identical classification tasks
    items = [
        AnnotationItem(
            item_id=f"item_{i}",
            annotator_id="worker_spammer",
            task_type=TaskType.CLASSIFICATION,
            label="positive",
            duration_seconds=8.0,
            confidence=0.95,
        )
        for i in range(6)
    ]
    defects = detector.scan_repetitive_patterns(items, max_consecutive_identical=5)
    assert len(defects) >= 5
    assert all(d.defect_type == DefectType.REPETITIVE_PATTERN for d in defects)
    assert all(d.annotator_id == "worker_spammer" for d in defects)


def test_repetitive_pattern_diverse_labels_not_flagged():
    detector = DefectDetector()
    labels = ["positive", "negative", "neutral", "positive", "negative"]
    items = [
        AnnotationItem(
            item_id=f"item_div_{i}",
            annotator_id="worker_honest",
            task_type=TaskType.CLASSIFICATION,
            label=lbl,
            duration_seconds=10.0,
            confidence=0.90,
        )
        for i, lbl in enumerate(labels)
    ]
    defects = detector.scan_repetitive_patterns(items, max_consecutive_identical=5)
    assert len(defects) == 0



