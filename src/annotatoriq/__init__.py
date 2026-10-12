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
from annotatoriq.defect_detector import DefectDetector
from annotatoriq.gold_eval import GoldEvaluator
from annotatoriq.metrics import AgreementMetrics
from annotatoriq.release_gate import QualityGate
from annotatoriq.adjudication import AdjudicationEngine
from annotatoriq.drift import DriftDetector
from annotatoriq.report_generator import ReportGenerator

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
    "DefectDetector",
    "GoldEvaluator",
    "AgreementMetrics",
    "QualityGate",
    "AdjudicationEngine",
    "DriftDetector",
    "ReportGenerator",
]
