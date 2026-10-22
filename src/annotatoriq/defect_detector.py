"""
Defect Detection & Rule Validation Engine.
Identifies malformed boundaries, out-of-vocab labels, speed anomalies, and consensus disputes.
"""

from typing import List, Dict, Any, Set, Optional
from annotatoriq.schema import (
    AnnotationItem,
    DefectRecord,
    DefectType,
    TaskType,
)


class DefectDetector:
    """Automated QA scanner for catching labeling defects and worker anomalies."""

    def __init__(self, allowed_labels: Optional[Dict[TaskType, Set[str]]] = None):
        self.allowed_labels = allowed_labels or {
            TaskType.CLASSIFICATION: {"positive", "negative", "neutral", "ambiguous", "toxic", "safe"},
            TaskType.SPAN_NER: {"PER", "ORG", "LOC", "DATE", "DIAGNOSIS", "MEDICATION"},
            TaskType.BOUNDING_BOX: {"document_table", "chart_legend", "signature", "stamp", "paragraph"},
            TaskType.AUDIO_SEGMENT: {"speech", "silence", "noise", "applause"},
        }

    def scan_annotations(
        self,
        annotations: List[AnnotationItem],
        min_duration_sec: float = 2.0,
        min_confidence: float = 0.20,
        check_repetitive: bool = False,
        max_consecutive_identical: int = 5,
    ) -> List[DefectRecord]:
        defects: List[DefectRecord] = []

        for item in annotations:
            # 1. Speed Anomaly
            if item.duration_seconds < min_duration_sec:
                defects.append(
                    DefectRecord(
                        item_id=item.item_id,
                        annotator_id=item.annotator_id,
                        defect_type=DefectType.SPEED_ANOMALY,
                        description=f"Task completed in {item.duration_seconds:.1f}s (below {min_duration_sec:.1f}s threshold).",
                        severity="MEDIUM",
                        observed_value=item.duration_seconds,
                        expected_value=f">= {min_duration_sec}s",
                    )
                )

            # 2. Confidence Anomaly
            if item.confidence is not None and item.confidence < min_confidence:
                defects.append(
                    DefectRecord(
                        item_id=item.item_id,
                        annotator_id=item.annotator_id,
                        defect_type=DefectType.CONFIDENCE_ANOMALY,
                        description=f"Suspiciously low self-reported confidence ({item.confidence:.2f} < {min_confidence:.2f}).",
                        severity="MEDIUM",
                        observed_value=item.confidence,
                        expected_value=f">= {min_confidence}",
                    )
                )

            # 2. Classification checks
            if item.task_type == TaskType.CLASSIFICATION:
                if not item.label:
                    defects.append(
                        DefectRecord(
                            item_id=item.item_id,
                            annotator_id=item.annotator_id,
                            defect_type=DefectType.SCHEMA_INVALID,
                            description="Missing mandatory label for classification task.",
                            severity="CRITICAL",
                        )
                    )
                elif (
                    TaskType.CLASSIFICATION in self.allowed_labels
                    and item.label not in self.allowed_labels[TaskType.CLASSIFICATION]
                ):
                    defects.append(
                        DefectRecord(
                            item_id=item.item_id,
                            annotator_id=item.annotator_id,
                            defect_type=DefectType.OUT_OF_VOCAB,
                            description=f"Label '{item.label}' is not in approved taxonomy.",
                            severity="HIGH",
                            observed_value=item.label,
                            expected_value=list(self.allowed_labels[TaskType.CLASSIFICATION]),
                        )
                    )

            # 3. Span NER checks
            elif item.task_type == TaskType.SPAN_NER:
                if item.spans is None:
                    defects.append(
                        DefectRecord(
                            item_id=item.item_id,
                            annotator_id=item.annotator_id,
                            defect_type=DefectType.SCHEMA_INVALID,
                            description="Spans field is None for span_ner task.",
                            severity="CRITICAL",
                        )
                    )
                else:
                    text_len = len(item.input_text or "")
                    for span in item.spans:
                        if span.start >= span.end:
                            defects.append(
                                DefectRecord(
                                    item_id=item.item_id,
                                    annotator_id=item.annotator_id,
                                    defect_type=DefectType.BOUNDARY_INVERSION,
                                    description=f"Inverted span boundary: start {span.start} >= end {span.end}.",
                                    severity="CRITICAL",
                                    observed_value=[span.start, span.end],
                                )
                            )
                        if text_len > 0 and (span.start > text_len or span.end > text_len):
                            defects.append(
                                DefectRecord(
                                    item_id=item.item_id,
                                    annotator_id=item.annotator_id,
                                    defect_type=DefectType.OUT_OF_BOUNDS,
                                    description=f"Span [{span.start}:{span.end}] exceeds text length ({text_len}).",
                                    severity="CRITICAL",
                                    observed_value=[span.start, span.end],
                                    expected_value=f"<= {text_len}",
                                )
                            )
                        if (
                            TaskType.SPAN_NER in self.allowed_labels
                            and span.label not in self.allowed_labels[TaskType.SPAN_NER]
                        ):
                            defects.append(
                                DefectRecord(
                                    item_id=item.item_id,
                                    annotator_id=item.annotator_id,
                                    defect_type=DefectType.OUT_OF_VOCAB,
                                    description=f"Span entity label '{span.label}' not in taxonomy.",
                                    severity="HIGH",
                                    observed_value=span.label,
                                )
                            )

            # 4. Bounding Box checks
            elif item.task_type == TaskType.BOUNDING_BOX:
                if item.boxes is None:
                    defects.append(
                        DefectRecord(
                            item_id=item.item_id,
                            annotator_id=item.annotator_id,
                            defect_type=DefectType.SCHEMA_INVALID,
                            description="Boxes field is None for bounding_box task.",
                            severity="CRITICAL",
                        )
                    )
                else:
                    for b in item.boxes:
                        if b.xmin >= b.xmax or b.ymin >= b.ymax:
                            defects.append(
                                DefectRecord(
                                    item_id=item.item_id,
                                    annotator_id=item.annotator_id,
                                    defect_type=DefectType.BOUNDARY_INVERSION,
                                    description=f"Inverted coordinates: xmin={b.xmin}>=xmax={b.xmax} or ymin={b.ymin}>=ymax={b.ymax}",
                                    severity="CRITICAL",
                                )
                            )
                        if any(c < 0.0 or c > 1.0 for c in [b.xmin, b.ymin, b.xmax, b.ymax]):
                            defects.append(
                                DefectRecord(
                                    item_id=item.item_id,
                                    annotator_id=item.annotator_id,
                                    defect_type=DefectType.OUT_OF_BOUNDS,
                                    description="Box coordinates out of normalized range [0.0, 1.0].",
                                    severity="CRITICAL",
                                )
                            )

            # 5. Audio Segment checks
            elif item.task_type == TaskType.AUDIO_SEGMENT:
                if item.segments is None:
                    defects.append(
                        DefectRecord(
                            item_id=item.item_id,
                            annotator_id=item.annotator_id,
                            defect_type=DefectType.SCHEMA_INVALID,
                            description="Segments field is None for audio_segment task.",
                            severity="CRITICAL",
                        )
                    )
                else:
                    for seg in item.segments:
                        if seg.start_sec >= seg.end_sec:
                            defects.append(
                                DefectRecord(
                                    item_id=item.item_id,
                                    annotator_id=item.annotator_id,
                                    defect_type=DefectType.BOUNDARY_INVERSION,
                                    description=f"Inverted audio segment: start_sec {seg.start_sec} >= end_sec {seg.end_sec}.",
                                    severity="CRITICAL",
                                    observed_value=[seg.start_sec, seg.end_sec],
                                )
                            )

        if check_repetitive:
            defects.extend(
                self.scan_repetitive_patterns(
                    annotations, max_consecutive_identical=max_consecutive_identical
                )
            )

        return defects

    def scan_repetitive_patterns(
        self,
        annotations: List[AnnotationItem],
        max_consecutive_identical: int = 5,
    ) -> List[DefectRecord]:
        """
        Detects straight-lining or click-farming behavior where an annotator
        repeatedly submits identical responses across consecutive items.
        """
        from collections import defaultdict

        grouped: Dict[str, List[AnnotationItem]] = defaultdict(list)
        for item in annotations:
            grouped[item.annotator_id].append(item)

        defects: List[DefectRecord] = []
        for ann_id, items in grouped.items():
            if len(items) < max_consecutive_identical:
                continue

            current_label = None
            run_items: List[AnnotationItem] = []

            for item in items:
                # Determine normalized label representation across task types
                label_repr = item.label
                if label_repr is None and item.spans:
                    label_repr = ";".join(f"{s.start}:{s.end}:{s.label}" for s in item.spans)
                elif label_repr is None and item.boxes:
                    label_repr = ";".join(f"{b.label}:{b.xmin:.2f}:{b.ymin:.2f}" for b in item.boxes)
                elif label_repr is None and item.segments:
                    label_repr = ";".join(f"{seg.start_sec:.1f}:{seg.end_sec:.1f}" for seg in item.segments)

                if label_repr is not None and label_repr == current_label:
                    run_items.append(item)
                else:
                    if len(run_items) >= max_consecutive_identical:
                        for flagged in run_items:
                            defects.append(
                                DefectRecord(
                                    item_id=flagged.item_id,
                                    annotator_id=ann_id,
                                    defect_type=DefectType.REPETITIVE_PATTERN,
                                    description=(
                                        f"Straight-lining defect: {len(run_items)} consecutive identical "
                                        f"labels ('{current_label}') submitted by {ann_id}."
                                    ),
                                    severity="HIGH",
                                    observed_value=current_label,
                                )
                            )
                    current_label = label_repr
                    run_items = [item] if label_repr is not None else []

            # Check trailing run
            if len(run_items) >= max_consecutive_identical:
                for flagged in run_items:
                    defects.append(
                        DefectRecord(
                            item_id=flagged.item_id,
                            annotator_id=ann_id,
                            defect_type=DefectType.REPETITIVE_PATTERN,
                            description=(
                                f"Straight-lining defect: {len(run_items)} consecutive identical "
                                f"labels ('{current_label}') submitted by {ann_id}."
                            ),
                            severity="HIGH",
                            observed_value=current_label,
                        )
                    )

        return defects
