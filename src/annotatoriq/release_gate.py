"""
Quality Release Gating & Deployment Policy Engine.
Enforces statistical thresholds and defect limits before training data enters pipeline.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from annotatoriq.schema import (
    QualityGateResult,
    AnnotationItem,
    GoldItem,
    DefectRecord,
    AdjudicationRecord,
    TaskType,
)
from annotatoriq.metrics import AgreementMetrics
from annotatoriq.gold_eval import GoldEvaluator
from annotatoriq.defect_detector import DefectDetector
from annotatoriq.adjudication import AdjudicationEngine


class QualityGate:
    """Enforces strict production quality standards on training datasets."""

    def __init__(
        self,
        min_krippendorff_alpha: float = 0.70,
        min_fleiss_kappa: float = 0.65,
        min_gold_accuracy: float = 0.85,
        max_unresolved_disputes: int = 0,
        max_critical_defects: int = 0,
    ):
        self.min_alpha = min_krippendorff_alpha
        self.min_kappa = min_fleiss_kappa
        self.min_gold_acc = min_gold_accuracy
        self.max_unresolved = max_unresolved_disputes
        self.max_critical_defects = max_critical_defects

    def evaluate_dataset(
        self,
        annotations: List[AnnotationItem],
        gold_items: Optional[List[GoldItem]] = None,
        adjudications: Optional[List[AdjudicationRecord]] = None,
    ) -> QualityGateResult:
        gold_items = gold_items or []
        adjudications = adjudications or []
        failure_reasons: List[str] = []

        # 1. Defect Scan
        detector = DefectDetector()
        defects = detector.scan_annotations(annotations)
        crit_defects = [d for d in defects if d.severity == "CRITICAL"]
        if len(crit_defects) > self.max_critical_defects:
            failure_reasons.append(
                f"Critical defect count ({len(crit_defects)}) exceeds permitted maximum ({self.max_critical_defects})."
            )

        # 2. Inter-annotator agreement on classification items
        clf_items = [a for a in annotations if a.task_type == TaskType.CLASSIFICATION]
        item_groups: Dict[str, List[AnnotationItem]] = {}
        for a in clf_items:
            item_groups.setdefault(a.item_id, []).append(a)

        multi_rated = [items for items in item_groups.values() if len(items) > 1]
        all_labels = sorted(list(set(a.label for a in clf_items if a.label)))
        label_map = {lbl: idx for idx, lbl in enumerate(all_labels)}
        K = len(all_labels)

        alpha_val = 1.0
        kappa_val = 1.0

        if multi_rated and K > 1:
            matrix = np.zeros((len(multi_rated), K), dtype=int)
            for r_idx, items in enumerate(multi_rated):
                for a in items:
                    if a.label in label_map:
                        matrix[r_idx, label_map[a.label]] += 1

            alpha_val = AgreementMetrics.compute_krippendorff_alpha_nominal(matrix)
            kappa_val = AgreementMetrics.compute_fleiss_kappa(matrix)

            if alpha_val < self.min_alpha:
                failure_reasons.append(
                    f"Krippendorff's Alpha ({alpha_val:.3f}) is below minimum threshold ({self.min_alpha:.2f})."
                )
            if kappa_val < self.min_kappa:
                failure_reasons.append(
                    f"Fleiss' Kappa ({kappa_val:.3f}) is below minimum threshold ({self.min_kappa:.2f})."
                )

        # 3. Gold standard evaluation
        gold_acc = 1.0
        if gold_items:
            eval_summary = GoldEvaluator.evaluate_annotators(annotations, gold_items)
            if eval_summary:
                avg_acc = np.mean([v["gold_accuracy"] for v in eval_summary.values()])
                gold_acc = float(avg_acc)
                if gold_acc < self.min_gold_acc:
                    failure_reasons.append(
                        f"Average Gold Standard Accuracy ({gold_acc*100:.1f}%) is below minimum ({self.min_gold_acc*100:.1f}%)."
                    )

        # 4. Dispute Resolution
        disputes = AdjudicationEngine.identify_disputes(annotations)
        adj_ids = {a.item_id for a in adjudications}
        unresolved = [d for d in disputes if d["item_id"] not in adj_ids]
        unresolved_pct = (len(unresolved) / len(item_groups) * 100) if item_groups else 0.0

        if len(unresolved) > self.max_unresolved:
            failure_reasons.append(
                f"Unresolved dispute count ({len(unresolved)}) exceeds maximum allowed ({self.max_unresolved})."
            )

        passed = (len(failure_reasons) == 0)

        return QualityGateResult(
            passed=passed,
            krippendorff_alpha=round(alpha_val, 4),
            fleiss_kappa=round(kappa_val, 4),
            gold_accuracy=round(gold_acc, 4),
            unresolved_dispute_count=len(unresolved),
            unresolved_dispute_pct=round(unresolved_pct, 2),
            defect_count=len(defects),
            gate_thresholds={
                "min_alpha": self.min_alpha,
                "min_kappa": self.min_kappa,
                "min_gold_accuracy": self.min_gold_acc,
                "max_unresolved_disputes": self.max_unresolved,
                "max_critical_defects": self.max_critical_defects,
            },
            failure_reasons=failure_reasons,
        )
