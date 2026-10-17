"""
Unit tests for AnnotatorIQ Drift Detection, Entropy Collapse, and Fatigue Indexing.
"""

import numpy as np
import pytest
from annotatoriq.schema import AnnotationItem, TaskType
from annotatoriq.drift import DriftDetector
from annotatoriq.release_gate import QualityGate


def test_compute_entropy():
    # Uniform 4-class distribution has maximum entropy: log2(4) = 2.0
    uniform = np.array([25, 25, 25, 25])
    ent_uniform = DriftDetector.compute_entropy(uniform)
    assert 1.95 <= ent_uniform <= 2.05

    # Completely skewed (single class) has near 0 entropy
    skewed = np.array([100, 0, 0, 0])
    ent_skewed = DriftDetector.compute_entropy(skewed)
    assert ent_skewed < 0.05


def test_detect_entropy_collapse():
    # Annotator a1 labels balanced at first, but then spams 'positive' for 15 items
    items = []
    # Normal diverse start
    for i in range(10):
        lbl = "positive" if i % 2 == 0 else "negative"
        items.append(
            AnnotationItem(
                item_id=f"item_{i}",
                annotator_id="a1",
                task_type=TaskType.CLASSIFICATION,
                label=lbl,
            )
        )
    # Collapsed run (botting / extreme fatigue)
    for i in range(10, 25):
        items.append(
            AnnotationItem(
                item_id=f"item_{i}",
                annotator_id="a1",
                task_type=TaskType.CLASSIFICATION,
                label="positive",
            )
        )

    defects = DriftDetector.detect_entropy_collapse(items, window_size=10, min_entropy=0.30)
    assert len(defects) >= 1
    assert any("entropy collapse" in d.description.lower() for d in defects)


def test_annotator_fatigue_index():
    # Annotator a1 gets rushed: 1st half 30s/item, 2nd half 3s/item
    items = []
    for i in range(10):
        lbl = "pos" if i % 2 == 0 else "neg"
        items.append(
            AnnotationItem(
                item_id=f"item_{i}",
                annotator_id="fatigued_worker",
                task_type=TaskType.CLASSIFICATION,
                label=lbl,
                duration_seconds=30.0,
            )
        )
    for i in range(10, 20):
        items.append(
            AnnotationItem(
                item_id=f"item_{i}",
                annotator_id="fatigued_worker",
                task_type=TaskType.CLASSIFICATION,
                label="pos",
                duration_seconds=3.0,
            )
        )

    profiles = DriftDetector.compute_annotator_fatigue_index(items)
    assert "fatigued_worker" in profiles
    profile = profiles["fatigued_worker"]
    assert profile["fatigue_score"] > 0.50
    assert profile["speed_drop_ratio"] > 0.80
    assert "RECOMMENDED" in profile["status"] or "DRIFT" in profile["status"]


def test_quality_gate_drift_enforcement():
    # Items with severe entropy collapse
    items = [
        AnnotationItem(item_id=f"item_{i}", annotator_id="bot1", task_type=TaskType.CLASSIFICATION, label="pos", duration_seconds=1.0)
        for i in range(25)
    ]
    # Add a couple neutral to create multi-class set
    items.append(AnnotationItem(item_id="item_0", annotator_id="bot1", task_type=TaskType.CLASSIFICATION, label="neg", duration_seconds=1.0))

    # Strict gate: max_drift_defects=0
    strict_gate = QualityGate(max_drift_defects=0)
    res = strict_gate.evaluate_dataset(items)
    assert res.passed is False
    assert any("Drift & entropy collapse" in r for r in res.failure_reasons)
