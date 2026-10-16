"""
Statistical Agreement & Multi-Task Metric Computation Engine.
Computes Cohen's Kappa, Fleiss' Kappa, Krippendorff's Alpha, Token F1, IoU, and Word Error Rate.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from annotatoriq.schema import SpanEntity, BoundingBox, AudioSegment


class AgreementMetrics:
    """Enterprise metrics suite for classification, spans, bounding boxes, and audio."""

    @staticmethod
    def compute_cohens_kappa(r1: List[str], r2: List[str]) -> float:
        """Computes Cohen's Kappa for two annotators across N items."""
        if len(r1) != len(r2) or len(r1) == 0:
            return 0.0

        categories = sorted(list(set(r1).union(set(r2))))
        k = len(categories)
        if k <= 1:
            return 1.0

        cat_map = {c: i for i, c in enumerate(categories)}
        cm = np.zeros((k, k), dtype=int)
        for a, b in zip(r1, r2):
            cm[cat_map[a], cat_map[b]] += 1

        total = np.sum(cm)
        if total == 0:
            return 0.0

        po = np.trace(cm) / total
        row_marginals = np.sum(cm, axis=1) / total
        col_marginals = np.sum(cm, axis=0) / total
        pe = np.sum(row_marginals * col_marginals)

        if pe == 1.0:
            return 1.0
        return float(np.clip((po - pe) / (1.0 - pe), -1.0, 1.0))

    @staticmethod
    def compute_fleiss_kappa(matrix: np.ndarray) -> float:
        """
        Computes Fleiss' Kappa for inter-annotator agreement.
        Args:
            matrix: N x K matrix (N items, K categories) with count of annotators assigning category k.
        """
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            return 0.0

        N, K = matrix.shape
        n_raters = np.sum(matrix[0, :])
        if n_raters <= 1 or N == 0:
            return 1.0

        p = np.sum(matrix, axis=0) / (N * n_raters)
        P_i = (np.sum(matrix**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
        P_bar = np.mean(P_i)
        P_e_bar = np.sum(p**2)

        if np.isclose(P_e_bar, 1.0):
            return 1.0

        kappa = (P_bar - P_e_bar) / (1.0 - P_e_bar)
        return float(np.clip(kappa, -1.0, 1.0))

    @staticmethod
    def compute_krippendorff_alpha_nominal(matrix: np.ndarray) -> float:
        """
        Computes Krippendorff's Alpha for nominal categories.
        Handles missing data / variable number of raters per item.
        """
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            return 0.0

        N, K = matrix.shape
        units_n = np.sum(matrix, axis=1)
        valid_units = units_n > 1
        if not np.any(valid_units):
            return 1.0

        total_pairs = np.sum(units_n[valid_units] * (units_n[valid_units] - 1))
        if total_pairs == 0:
            return 1.0

        coincidence = np.zeros((K, K))
        for i in range(N):
            if units_n[i] > 1:
                row = matrix[i, :]
                for c1 in range(K):
                    for c2 in range(K):
                        if c1 == c2:
                            coincidence[c1, c2] += row[c1] * (row[c1] - 1)
                        else:
                            coincidence[c1, c2] += row[c1] * row[c2]

        off_diag = np.sum(coincidence) - np.trace(coincidence)
        D_o = off_diag / total_pairs

        marginal = np.sum(coincidence, axis=1)
        total_coinc_pairs = np.sum(marginal)
        if total_coinc_pairs <= 1:
            return 1.0

        D_e = 0.0
        for c1 in range(K):
            for c2 in range(K):
                if c1 != c2:
                    D_e += (marginal[c1] * marginal[c2]) / (total_coinc_pairs * (total_coinc_pairs - 1))

        if D_e == 0:
            return 1.0

        alpha = 1.0 - (D_o / D_e)
        return float(np.clip(alpha, -1.0, 1.0))

    @staticmethod
    def bbox_iou(box1: BoundingBox, box2: BoundingBox) -> float:
        """Computes 2D Intersection-over-Union (IoU) for normalized bounding boxes."""
        if box1.label != box2.label:
            return 0.0

        x_inter_min = max(box1.xmin, box2.xmin)
        y_inter_min = max(box1.ymin, box2.ymin)
        x_inter_max = min(box1.xmax, box2.xmax)
        y_inter_max = min(box1.ymax, box2.ymax)

        inter_w = max(0.0, x_inter_max - x_inter_min)
        inter_h = max(0.0, y_inter_max - y_inter_min)
        inter_area = inter_w * inter_h

        area1 = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin)
        area2 = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin)
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0
        return float(inter_area / union_area)

    @staticmethod
    def span_char_iou(s1: SpanEntity, s2: SpanEntity) -> float:
        """Computes Character-level Intersection-over-Union for text spans."""
        if s1.label != s2.label:
            return 0.0

        start_inter = max(s1.start, s2.start)
        end_inter = min(s1.end, s2.end)
        inter_len = max(0, end_inter - start_inter)

        len1 = s1.end - s1.start
        len2 = s2.end - s2.start
        union_len = len1 + len2 - inter_len

        if union_len <= 0:
            return 0.0
        return float(inter_len / union_len)

    @staticmethod
    def span_token_f1(pred_spans: List[SpanEntity], gold_spans: List[SpanEntity]) -> Dict[str, float]:
        """Calculates Precision, Recall, and F1 over span entities."""
        if not pred_spans and not gold_spans:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if not pred_spans or not gold_spans:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        true_positives = 0
        used_gold = set()

        for pred in pred_spans:
            for g_idx, gold in enumerate(gold_spans):
                if g_idx not in used_gold:
                    # Match if label matches and character IoU >= 0.50
                    if pred.label == gold.label and AgreementMetrics.span_char_iou(pred, gold) >= 0.50:
                        true_positives += 1
                        used_gold.add(g_idx)
                        break

        prec = true_positives / len(pred_spans) if pred_spans else 0.0
        rec = true_positives / len(gold_spans) if gold_spans else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

    @staticmethod
    def word_error_rate(hypothesis: str, reference: str) -> float:
        """Computes Levenshtein Word Error Rate (WER) for transcription."""
        hyp_words = hypothesis.strip().split()
        ref_words = reference.strip().split()

        if not ref_words:
            return 0.0 if not hyp_words else 1.0
        if not hyp_words:
            return 1.0

        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)
        for i in range(len(ref_words) + 1):
            d[i, 0] = i
        for j in range(len(hyp_words) + 1):
            d[0, j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i - 1].lower() == hyp_words[j - 1].lower():
                    d[i, j] = d[i - 1, j - 1]
                else:
                    d[i, j] = min(
                        d[i - 1, j] + 1,      # deletion
                        d[i, j - 1] + 1,      # insertion
                        d[i - 1, j - 1] + 1,  # substitution
                    )

        wer = d[len(ref_words), len(hyp_words)] / len(ref_words)
        return float(min(1.0, wer))

    @staticmethod
    def bootstrap_agreement_ci(
        r1: List[str],
        r2: List[str],
        n_bootstraps: int = 200,
        ci: float = 0.95,
        seed: int = 42,
    ) -> Dict[str, float]:
        """
        Calculates empirical bootstrap confidence intervals for Cohen's Kappa.
        Returns point estimate, ci_lower, and ci_upper.
        """
        if len(r1) != len(r2) or len(r1) == 0:
            return {"point_estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std_error": 0.0}

        point_est = AgreementMetrics.compute_cohens_kappa(r1, r2)
        n = len(r1)
        rng = np.random.default_rng(seed)

        boot_scores = []
        r1_arr = np.array(r1)
        r2_arr = np.array(r2)

        for _ in range(n_bootstraps):
            indices = rng.integers(0, n, size=n)
            sample_r1 = r1_arr[indices].tolist()
            sample_r2 = r2_arr[indices].tolist()
            boot_scores.append(AgreementMetrics.compute_cohens_kappa(sample_r1, sample_r2))

        alpha = (1.0 - ci) / 2.0
        ci_lower = float(np.percentile(boot_scores, alpha * 100))
        ci_upper = float(np.percentile(boot_scores, (1.0 - alpha) * 100))

        return {
            "point_estimate": round(point_est, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "std_error": round(float(np.std(boot_scores)), 4),
        }

    @staticmethod
    def compute_brier_and_calibration(
        probabilities: List[float],
        outcomes: List[int],
    ) -> Dict[str, float]:
        """
        Computes the Brier score and mean calibration error for annotator confidence probabilities.
        """
        if len(probabilities) != len(outcomes) or len(outcomes) == 0:
            return {"brier_score": 0.0, "mean_calibration_error": 0.0}

        probs = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
        actual = np.asarray(outcomes, dtype=float)

        brier = float(np.mean((probs - actual) ** 2))
        calibration_error = float(np.abs(np.mean(probs) - np.mean(actual)))

        return {
            "brier_score": round(brier, 4),
            "mean_calibration_error": round(calibration_error, 4),
        }

