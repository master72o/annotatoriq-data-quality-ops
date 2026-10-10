"""
AnnotatorIQ: Enterprise Training Data Quality Operations & Governance Platform.
"""

__version__ = "2.0.0"
__author__ = "Munir (master72o)"

from annotatoriq.schema import (
    TaskType,
    DefectType,
    ReliabilityTier,
    SpanEntity,
    BoundingBox,
    AudioSegment,
    AnnotationItem,
    GoldItem,
    DefectRecord,
    AdjudicationRecord,
    QualityGateResult,
)

__all__ = [
    "TaskType",
    "DefectType",
    "ReliabilityTier",
    "SpanEntity",
    "BoundingBox",
    "AudioSegment",
    "AnnotationItem",
    "GoldItem",
    "DefectRecord",
    "AdjudicationRecord",
    "QualityGateResult",
]
