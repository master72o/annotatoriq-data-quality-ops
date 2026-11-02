"""
Adjudication Engine & Immutable Audit Trail Logger.
Handles dispute routing, human expert adjudication, and production release exports.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from collections import Counter
from annotatoriq.schema import (
    AnnotationItem,
    AdjudicationRecord,
    TaskType,
)


class AdjudicationEngine:
    """Manages the lifecycle of disputed annotations and final ground-truth resolution."""

    @staticmethod
    def identify_disputes(annotations: List[AnnotationItem]) -> List[Dict[str, Any]]:
        """Groups annotations by item_id and flags items with inter-annotator disagreement."""
        grouped: Dict[str, List[AnnotationItem]] = {}
        for a in annotations:
            grouped.setdefault(a.item_id, []).append(a)

        disputes: List[Dict[str, Any]] = []
        for item_id, items in grouped.items():
            if len(items) <= 1:
                continue

            task_type = items[0].task_type
            if task_type == TaskType.CLASSIFICATION:
                labels = [i.label for i in items if i.label is not None]
                counts = Counter(labels)
                if len(counts) > 1:
                    disputes.append({
                        "item_id": item_id,
                        "task_type": task_type,
                        "input_text": items[0].input_text,
                        "annotations": [
                            {"annotator_id": i.annotator_id, "label": i.label, "confidence": i.confidence}
                            for i in items
                        ],
                        "vote_distribution": dict(counts),
                        "status": "UNRESOLVED",
                    })

            elif task_type == TaskType.SPAN_NER:
                span_counts = [len(i.spans or []) for i in items]
                if len(set(span_counts)) > 1:
                    disputes.append({
                        "item_id": item_id,
                        "task_type": task_type,
                        "input_text": items[0].input_text,
                        "annotations": [
                            {"annotator_id": i.annotator_id, "span_count": len(i.spans or [])}
                            for i in items
                        ],
                        "status": "UNRESOLVED",
                    })

        return disputes

    @staticmethod
    def resolve_dispute(
        item_id: str,
        task_type: TaskType,
        disputed_options: List[Dict[str, Any]],
        final_decision: Dict[str, Any],
        adjudicator_id: str,
        rationale: str,
        confidence: float = 1.0,
    ) -> AdjudicationRecord:
        """Applies an expert adjudication decision and logs timestamped rationale."""
        return AdjudicationRecord(
            item_id=item_id,
            task_type=task_type,
            disputed_options=disputed_options,
            final_decision=final_decision,
            adjudicator_id=adjudicator_id,
            rationale=rationale,
            confidence=confidence,
            resolved_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def export_adjudicated_dataset(
        raw_annotations: List[AnnotationItem],
        adjudications: List[AdjudicationRecord],
    ) -> List[Dict[str, Any]]:
        """Produces a clean, verified dataset replacing disputed items with expert decisions."""
        adj_map = {a.item_id: a for a in adjudications}
        seen_items = set()
        release_items: List[Dict[str, Any]] = []

        for ann in raw_annotations:
            if ann.item_id in seen_items:
                continue

            seen_items.add(ann.item_id)
            if ann.item_id in adj_map:
                adj = adj_map[ann.item_id]
                release_items.append({
                    "item_id": ann.item_id,
                    "task_type": ann.task_type.value,
                    "input_text": ann.input_text,
                    "final_label": adj.final_decision.get("label"),
                    "final_spans": adj.final_decision.get("spans"),
                    "provenance": "ADJUDICATED_EXPERT",
                    "adjudicated_by": adj.adjudicator_id,
                    "rationale": adj.rationale,
                    "resolved_at": adj.resolved_at,
                })
            else:
                release_items.append({
                    "item_id": ann.item_id,
                    "task_type": ann.task_type.value,
                    "input_text": ann.input_text,
                    "final_label": ann.label,
                    "final_spans": [s.dict() for s in (ann.spans or [])],
                    "provenance": "CONSENSUS_DIRECT",
                })

        return release_items

    @staticmethod
    def compute_confidence_weighted_consensus(
        annotations: List[AnnotationItem],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Computes confidence-weighted majority consensus per item_id.
        Returns winning label, weighted vote share, margin of victory, and whether
        expert escalation is recommended (margin < 0.25).
        """
        grouped: Dict[str, List[AnnotationItem]] = {}
        for ann in annotations:
            if ann.task_type == TaskType.CLASSIFICATION and ann.label is not None:
                grouped.setdefault(ann.item_id, []).append(ann)

        results: Dict[str, Dict[str, Any]] = {}
        for item_id, items in grouped.items():
            label_weights: Dict[str, float] = {}
            total_weight = 0.0
            for i in items:
                w = max(float(i.confidence if i.confidence is not None else 1.0), 0.05)
                label_weights[i.label] = label_weights.get(i.label, 0.0) + w
                total_weight += w

            sorted_labels = sorted(label_weights.items(), key=lambda x: x[1], reverse=True)
            winner_label, winner_weight = sorted_labels[0]
            runner_up_weight = sorted_labels[1][1] if len(sorted_labels) > 1 else 0.0

            winner_share = winner_weight / total_weight if total_weight > 0 else 0.0
            margin = (winner_weight - runner_up_weight) / total_weight if total_weight > 0 else 1.0

            results[item_id] = {
                "consensus_label": winner_label,
                "weighted_share": round(winner_share, 4),
                "margin_of_victory": round(margin, 4),
                "requires_escalation": margin < 0.25,
                "annotator_count": len(items),
            }

        return results

    @staticmethod
    def audit_adjudicator_overturn_rates(
        raw_annotations: List[AnnotationItem],
        adjudications: List[AdjudicationRecord],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Audits how often each annotator's labels were overturned by expert adjudicators
        on disputed items, surfacing systematic worker misalignment.
        """
        adj_map = {a.item_id: a for a in adjudications}
        worker_stats: Dict[str, Dict[str, int]] = {}

        for ann in raw_annotations:
            if ann.item_id not in adj_map or ann.task_type != TaskType.CLASSIFICATION:
                continue
            expert_label = adj_map[ann.item_id].final_decision.get("label")
            if expert_label is None:
                continue

            stats = worker_stats.setdefault(ann.annotator_id, {"adjudicated_items": 0, "overturned": 0})
            stats["adjudicated_items"] += 1
            if ann.label != expert_label:
                stats["overturned"] += 1

        report: Dict[str, Dict[str, Any]] = {}
        for worker_id, st in worker_stats.items():
            total = st["adjudicated_items"]
            overturned = st["overturned"]
            rate = overturned / total if total > 0 else 0.0
            report[worker_id] = {
                "adjudicated_items": total,
                "overturned_count": overturned,
                "overturn_rate": round(rate, 4),
                "calibration_flag": "RECALIBRATE" if rate > 0.50 else "ALIGNED",
            }

        return report

