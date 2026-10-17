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

    @staticmethod
    def compute_entropy(p: np.ndarray, epsilon: float = 1e-9) -> float:
        """Computes Shannon entropy (base 2) of a categorical distribution."""
        p = np.asarray(p, dtype=float) + epsilon
        p = p / np.sum(p)
        return float(-np.sum(p * np.log2(p)))

    @staticmethod
    def detect_entropy_collapse(
        annotations: List[AnnotationItem],
        window_size: int = 10,
        min_entropy: float = 0.20,
    ) -> List[DefectRecord]:
        """
        Detects repetitive single-class labeling or sudden entropy collapse
        over sliding windows of annotations (indicating botting or extreme fatigue).
        """
        defects: List[DefectRecord] = []
        by_annotator: Dict[str, List[AnnotationItem]] = {}

        for a in annotations:
            by_annotator.setdefault(a.annotator_id, []).append(a)

        for a_id, items in by_annotator.items():
            if len(items) < window_size:
                continue

            labels = sorted(list(set(i.label for i in items if i.label)))
            if len(labels) <= 1:
                continue

            label_map = {lbl: idx for idx, lbl in enumerate(labels)}
            k = len(labels)

            for start_idx in range(0, len(items) - window_size + 1, window_size // 2):
                win = items[start_idx : start_idx + window_size]
                counts = np.zeros(k)
                for item in win:
                    if item.label in label_map:
                        counts[label_map[item.label]] += 1

                entropy = DriftDetector.compute_entropy(counts)
                if entropy < min_entropy:
                    dominant_label = labels[int(np.argmax(counts))]
                    defects.append(
                        DefectRecord(
                            item_id=win[0].item_id,
                            annotator_id=a_id,
                            defect_type=DefectType.DRIFT_DETECTED,
                            description=(
                                f"Annotator {a_id} experienced entropy collapse in window "
                                f"[{start_idx}:{start_idx + window_size}] "
                                f"(Entropy: {entropy:.3f} < {min_entropy:.2f}, dominant label: '{dominant_label}')."
                            ),
                            severity="HIGH",
                            observed_value=round(entropy, 3),
                            expected_value=f">= {min_entropy}",
                        )
                    )
        return defects

    @staticmethod
    def compute_annotator_fatigue_index(
        annotations: List[AnnotationItem],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Computes annotator fatigue and pacing decay by analyzing duration decay
        and label repetition across halves of the session.
        """
        by_annotator: Dict[str, List[AnnotationItem]] = {}
        for a in annotations:
            by_annotator.setdefault(a.annotator_id, []).append(a)

        fatigue_profiles: Dict[str, Dict[str, Any]] = {}
        for a_id, items in by_annotator.items():
            if len(items) < 6:
                fatigue_profiles[a_id] = {
                    "sample_count": len(items),
                    "fatigue_score": 0.0,
                    "status": "INSUFFICIENT_HISTORY",
                }
                continue

            mid = len(items) // 2
            first_half = items[:mid]
            second_half = items[mid:]

            dur_first = [i.duration_seconds for i in first_half if i.duration_seconds is not None]
            dur_second = [i.duration_seconds for i in second_half if i.duration_seconds is not None]

            speed_drop_ratio = 0.0
            if dur_first and dur_second and np.median(dur_first) > 0:
                speed_drop_ratio = float((np.median(dur_first) - np.median(dur_second)) / np.median(dur_first))

            # If annotator rushed > 40% faster in 2nd half, high risk of fatigue clicking
            rush_penalty = max(0.0, speed_drop_ratio)

            # Measure label entropy decay
            labels = sorted(list(set(i.label for i in items if i.label)))
            entropy_decay = 0.0
            if len(labels) > 1:
                label_map = {lbl: idx for idx, lbl in enumerate(labels)}
                c1 = np.zeros(len(labels))
                c2 = np.zeros(len(labels))
                for i in first_half:
                    if i.label in label_map:
                        c1[label_map[i.label]] += 1
                for i in second_half:
                    if i.label in label_map:
                        c2[label_map[i.label]] += 1
                e1 = DriftDetector.compute_entropy(c1)
                e2 = DriftDetector.compute_entropy(c2)
                if e1 > 0:
                    entropy_decay = max(0.0, (e1 - e2) / e1)

            fatigue_score = float(np.clip(0.6 * rush_penalty + 0.4 * entropy_decay, 0.0, 1.0))
            status = "HEALTHY"
            if fatigue_score > 0.65:
                status = "FATIGUED_BREAK_RECOMMENDED"
            elif fatigue_score > 0.40:
                status = "MODERATE_PACING_DRIFT"

            fatigue_profiles[a_id] = {
                "sample_count": len(items),
                "fatigue_score": round(fatigue_score, 3),
                "speed_drop_ratio": round(speed_drop_ratio, 3),
                "entropy_decay": round(entropy_decay, 3),
                "status": status,
            }

        return fatigue_profiles

