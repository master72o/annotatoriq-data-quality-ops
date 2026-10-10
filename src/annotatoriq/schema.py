"""
Extensible, strongly-typed schemas for Multi-Task Annotation & Quality Operations.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    SPAN_NER = "span_ner"
    BOUNDING_BOX = "bounding_box"
    AUDIO_SEGMENT = "audio_segment"


class DefectType(str, Enum):
    SCHEMA_INVALID = "schema_invalid"
    OUT_OF_VOCAB = "out_of_vocab"
    BOUNDARY_INVERSION = "boundary_inversion"
    OUT_OF_BOUNDS = "out_of_bounds"
    SPEED_ANOMALY = "speed_anomaly"
    CONSENSUS_DISAGREEMENT = "consensus_disagreement"
    DRIFT_DETECTED = "drift_detected"
    GOLD_FAILURE = "gold_failure"


class ReliabilityTier(str, Enum):
    TIER_1_EXPERT = "Tier 1 (Expert >= 90%)"
    TIER_2_RELIABLE = "Tier 2 (Reliable 80-89%)"
    TIER_3_NEEDS_TRAINING = "Tier 3 (Needs Training 70-79%)"
    TIER_4_FLAGGED = "Tier 4 (Flagged < 70%)"


class SpanEntity(BaseModel):
    label: str
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    text: Optional[str] = None

    @model_validator(mode="after")
    def validate_boundary(self):
        if self.start >= self.end:
            raise ValueError(f"Span start ({self.start}) must be strictly less than end ({self.end})")
        return self


class BoundingBox(BaseModel):
    label: str
    xmin: float = Field(..., ge=0.0, le=1.0)
    ymin: float = Field(..., ge=0.0, le=1.0)
    xmax: float = Field(..., ge=0.0, le=1.0)
    ymax: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_box(self):
        if self.xmin >= self.xmax:
            raise ValueError(f"xmin ({self.xmin}) must be strictly less than xmax ({self.xmax})")
        if self.ymin >= self.ymax:
            raise ValueError(f"ymin ({self.ymin}) must be strictly less than ymax ({self.ymax})")
        return self


class AudioSegment(BaseModel):
    start_sec: float = Field(..., ge=0.0)
    end_sec: float = Field(..., ge=0.0)
    transcript: str

    @model_validator(mode="after")
    def validate_audio(self):
        if self.start_sec >= self.end_sec:
            raise ValueError(f"start_sec ({self.start_sec}) must be strictly less than end_sec ({self.end_sec})")
        return self


class AnnotationItem(BaseModel):
    item_id: str
    annotator_id: str
    task_type: TaskType
    input_text: Optional[str] = None
    input_uri: Optional[str] = None
    label: Optional[str] = None
    spans: Optional[List[SpanEntity]] = None
    boxes: Optional[List[BoundingBox]] = None
    segments: Optional[List[AudioSegment]] = None
    duration_seconds: float = Field(default=15.0, ge=0.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GoldItem(BaseModel):
    item_id: str
    task_type: TaskType
    expected_label: Optional[str] = None
    expected_spans: Optional[List[SpanEntity]] = None
    expected_boxes: Optional[List[BoundingBox]] = None
    expected_segments: Optional[List[AudioSegment]] = None
    domain: Optional[str] = "general"
    rationale: Optional[str] = None


class DefectRecord(BaseModel):
    item_id: str
    annotator_id: str
    defect_type: DefectType
    description: str
    severity: str = "HIGH"  # CRITICAL, HIGH, MEDIUM, LOW
    observed_value: Any = None
    expected_value: Any = None


class AdjudicationRecord(BaseModel):
    item_id: str
    task_type: TaskType
    disputed_options: List[Dict[str, Any]]
    final_decision: Dict[str, Any]
    adjudicator_id: str
    rationale: str
    confidence: float = 1.0
    resolved_at: Optional[str] = None


class QualityGateResult(BaseModel):
    passed: bool
    krippendorff_alpha: float
    fleiss_kappa: float
    gold_accuracy: float
    unresolved_dispute_count: int
    unresolved_dispute_pct: float
    defect_count: int
    gate_thresholds: Dict[str, Any]
    failure_reasons: List[str] = Field(default_factory=list)
