"""
Annotator Drift & Distribution Shift Detection Engine.
Calculates Jensen-Shannon divergence and rolling accuracy across time windows.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from annotatoriq.schema import AnnotationItem, DefectRecord, DefectType


class DriftDetector:
    """Monitors annotators for class skew and quality drift over time windows."""

    @staticmethod
    def compute_js_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-9) -> float:
        """Computes Jensen-Shannon divergence between two categorical distributions."""
        p = np.asarray(p, dtype=float) + epsilon
        q = np.asarray(q, dtype=float) + epsilon
        p = p / np.sum(p)
        q = q / np.sum(q)

        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log2(p / m))
        kl_qm = np.sum(q * np.log2(q / m))
        js = 0.5 * (kl_pm + kl_qm)
        return float(np.clip(np.sqrt(max(0.0, js)), 0.0, 1.0))

    @staticmethod
    def detect_annotator_drift(
        annotations: List[AnnotationItem],
        batch_size: int = 10,
        drift_threshold: float = 0.25,
    ) -> List[DefectRecord]:
        """Detects distribution drift between first batch (baseline) and subsequent batches."""
        defects: List[DefectRecord] = []
        by_annotator: Dict[str, List[AnnotationItem]] = {}

        for a in annotations:
            by_annotator.setdefault(a.annotator_id, []).append(a)

        for a_id, items in by_annotator.items():
            if len(items) < batch_size * 2:
                continue

            labels = sorted(list(set(i.label for i in items if i.label)))
            if len(labels) <= 1:
                continue

            label_map = {lbl: idx for idx, lbl in enumerate(labels)}
            k = len(labels)

            # Baseline batch (first N items)
            base_items = items[:batch_size]
            base_counts = np.zeros(k)
            for i in base_items:
                if i.label in label_map:
                    base_counts[label_map[i.label]] += 1

            # Subsequent batches
            for b_idx in range(1, len(items) // batch_size):
                sub_items = items[b_idx * batch_size : (b_idx + 1) * batch_size]
                sub_counts = np.zeros(k)
                for i in sub_items:
                    if i.label in label_map:
                        sub_counts[label_map[i.label]] += 1

                js_div = DriftDetector.compute_js_divergence(base_counts, sub_counts)
                if js_div > drift_threshold:
                    defects.append(
                        DefectRecord(
                            item_id=sub_items[0].item_id,
                            annotator_id=a_id,
                            defect_type=DefectType.DRIFT_DETECTED,
                            description=(
                                f"Annotator {a_id} class distribution drifted in batch {b_idx + 1} "
                                f"(JS Divergence: {js_div:.3f} > {drift_threshold:.2f})."
                            ),
                            severity="HIGH",
                            observed_value=round(js_div, 3),
                            expected_value=f"<= {drift_threshold}",
                        )
                    )

        return defects
