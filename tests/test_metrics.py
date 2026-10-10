"""
Unit tests for AnnotatorIQ agreement and multi-task metrics.
"""

import numpy as np
import pytest
from annotatoriq.metrics import AgreementMetrics
from annotatoriq.schema import SpanEntity, BoundingBox


def test_cohens_kappa_perfect_agreement():
    r1 = ["pos", "neg", "neu", "pos"]
    r2 = ["pos", "neg", "neu", "pos"]
    k = AgreementMetrics.compute_cohens_kappa(r1, r2)
    assert k == 1.0


def test_cohens_kappa_disagreement():
    r1 = ["pos", "neg", "neu", "pos"]
    r2 = ["neg", "pos", "pos", "neu"]
    k = AgreementMetrics.compute_cohens_kappa(r1, r2)
    assert k < 0.20


def test_fleiss_kappa_perfect():
    # 5 items, 3 categories, all 3 annotators chose category 0
    mat = np.array([
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 3],
        [3, 0, 0],
        [0, 3, 0],
    ])
    kappa = AgreementMetrics.compute_fleiss_kappa(mat)
    assert kappa == 1.0


def test_krippendorff_alpha_nominal_perfect():
    mat = np.array([
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 3],
    ])
    alpha = AgreementMetrics.compute_krippendorff_alpha_nominal(mat)
    assert alpha == 1.0


def test_bbox_iou_identical():
    b1 = BoundingBox(label="table", xmin=0.1, ymin=0.1, xmax=0.5, ymax=0.5)
    b2 = BoundingBox(label="table", xmin=0.1, ymin=0.1, xmax=0.5, ymax=0.5)
    assert AgreementMetrics.bbox_iou(b1, b2) == 1.0


def test_bbox_iou_disjoint():
    b1 = BoundingBox(label="table", xmin=0.1, ymin=0.1, xmax=0.3, ymax=0.3)
    b2 = BoundingBox(label="table", xmin=0.5, ymin=0.5, xmax=0.8, ymax=0.8)
    assert AgreementMetrics.bbox_iou(b1, b2) == 0.0


def test_span_token_f1_match():
    p = [SpanEntity(label="PER", start=0, end=10, text="John Smith")]
    g = [SpanEntity(label="PER", start=0, end=10, text="John Smith")]
    res = AgreementMetrics.span_token_f1(p, g)
    assert res["f1"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0


def test_word_error_rate():
    hyp = "The quick brown fox jumps"
    ref = "The fast brown fox jumps"
    wer = AgreementMetrics.word_error_rate(hyp, ref)
    assert 0.15 < wer < 0.25
