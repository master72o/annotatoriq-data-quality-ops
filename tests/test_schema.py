"""
Unit tests for AnnotatorIQ Pydantic schemas and boundary validators.
"""

import pytest
from annotatoriq.schema import (
    TaskType,
    DefectType,
    SpanEntity,
    BoundingBox,
    AudioSegment,
    AnnotationItem,
)


def test_span_entity_valid():
    span = SpanEntity(label="PER", start=0, end=10, text="John Doe")
    assert span.start == 0
    assert span.end == 10
    assert span.label == "PER"


def test_span_entity_inverted_raises():
    with pytest.raises(ValueError):
        SpanEntity(label="PER", start=15, end=10)


def test_bounding_box_valid():
    box = BoundingBox(label="table", xmin=0.1, ymin=0.2, xmax=0.8, ymax=0.9)
    assert box.xmin < box.xmax
    assert box.ymin < box.ymax


def test_bounding_box_inverted_raises():
    with pytest.raises(ValueError):
        BoundingBox(label="table", xmin=0.8, ymin=0.2, xmax=0.1, ymax=0.9)


def test_audio_segment_valid():
    seg = AudioSegment(start_sec=1.5, end_sec=4.2, transcript="Hello world")
    assert seg.start_sec < seg.end_sec


def test_audio_segment_inverted_raises():
    with pytest.raises(ValueError):
        AudioSegment(start_sec=5.0, end_sec=2.0, transcript="Invalid")


def test_annotation_item_classification():
    item = AnnotationItem(
        item_id="task_001",
        annotator_id="ann_01",
        task_type=TaskType.CLASSIFICATION,
        label="positive",
        duration_seconds=12.5,
    )
    assert item.task_type == TaskType.CLASSIFICATION
    assert item.label == "positive"
