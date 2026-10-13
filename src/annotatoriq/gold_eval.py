"""
Hidden Gold Standard Evaluation & Annotator Reliability Tiering Engine.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from annotatoriq.schema import (
    AnnotationItem,
    GoldItem,
    ReliabilityTier,
    TaskType,
)
from annotatoriq.metrics import AgreementMetrics


class GoldEvaluator:
    """Evaluates annotator performance against hidden gold standards."""

    @staticmethod
    def evaluate_annotators(
        annotations: List[AnnotationItem], gold_items: List[GoldItem]
    ) -> Dict[str, Dict[str, Any]]:
        gold_map = {g.item_id: g for g in gold_items}
        annotator_evals: Dict[str, Dict[str, Any]] = {}

        for ann in annotations:
            if ann.item_id not in gold_map:
                continue

            gold = gold_map[ann.item_id]
            a_id = ann.annotator_id

            if a_id not in annotator_evals:
                annotator_evals[a_id] = {
                    "gold_count": 0,
                    "correct_count": 0,
                    "conf_sum": 0.0,
                    "durations": [],
                }

            is_correct = False
            if ann.task_type == TaskType.CLASSIFICATION:
                is_correct = (ann.label == gold.expected_label)
            elif ann.task_type == TaskType.SPAN_NER and ann.spans and gold.expected_spans:
                f1_res = AgreementMetrics.span_token_f1(ann.spans, gold.expected_spans)
                is_correct = f1_res["f1"] >= 0.70
            elif ann.task_type == TaskType.BOUNDING_BOX and ann.boxes and gold.expected_boxes:
                if len(ann.boxes) == len(gold.expected_boxes):
                    ious = [
                        AgreementMetrics.bbox_iou(b, gb)
                        for b, gb in zip(ann.boxes, gold.expected_boxes)
                    ]
                    is_correct = np.mean(ious) >= 0.60
            elif ann.task_type == TaskType.AUDIO_SEGMENT and ann.segments and gold.expected_segments:
                if len(ann.segments) == len(gold.expected_segments):
                    wers = [
                        AgreementMetrics.word_error_rate(s.transcript, gs.transcript)
                        for s, gs in zip(ann.segments, gold.expected_segments)
                    ]
                    is_correct = np.mean(wers) <= 0.15

            annotator_evals[a_id]["gold_count"] += 1
            if is_correct:
                annotator_evals[a_id]["correct_count"] += 1
            annotator_evals[a_id]["conf_sum"] += ann.confidence
            annotator_evals[a_id]["durations"].append(ann.duration_seconds)

        # Compute summary scores & tier
        summary: Dict[str, Dict[str, Any]] = {}
        for a_id, data in annotator_evals.items():
            count = data["gold_count"]
            acc = data["correct_count"] / count if count > 0 else 0.0
            avg_conf = data["conf_sum"] / count if count > 0 else 0.0
            avg_dur = float(np.mean(data["durations"])) if data["durations"] else 0.0

            if acc >= 0.90:
                tier = ReliabilityTier.TIER_1_EXPERT
            elif acc >= 0.80:
                tier = ReliabilityTier.TIER_2_RELIABLE
            elif acc >= 0.70:
                tier = ReliabilityTier.TIER_3_NEEDS_TRAINING
            else:
                tier = ReliabilityTier.TIER_4_FLAGGED

            summary[a_id] = {
                "gold_items_evaluated": count,
                "gold_accuracy": round(acc, 4),
                "gold_accuracy_pct": round(acc * 100, 1),
                "avg_confidence": round(avg_conf, 3),
                "avg_duration_sec": round(avg_dur, 2),
                "reliability_tier": tier.value,
            }

        return summary
