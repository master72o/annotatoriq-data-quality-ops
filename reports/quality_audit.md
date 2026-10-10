# 🛡️ AnnotatorIQ Data Quality Audit Report

**Dataset Batch:** `production_annotation_batch_v2`  
**Evaluation Timestamp:** `2026-10-10 16:51:36 UTC`  
**Gate Decision:** 🔴 **FAILED QUALITY GATES**  

---

## 1. Executive Quality Summary

| Metric | Measured Value | Policy Threshold | Status |
| :--- | :---: | :---: | :---: |
| **Krippendorff's Alpha** | `0.6546` | `>= 0.70` | ❌ Fail |
| **Fleiss' Kappa** | `0.3209` | `>= 0.65` | ❌ Fail |
| **Gold Standard Accuracy** | `87.5%` | `>= 85.0%` | ✅ Pass |
| **Unresolved Disputes** | `5` (`38.5%`) | `<= 0` | ❌ Fail |
| **Total Defects Flagged** | `4` | — | ⚠️ Review Required |

### ⚠️ Active Release Blockers

- ❌ Krippendorff's Alpha (0.655) is below minimum threshold (0.70).
- ❌ Fleiss' Kappa (0.321) is below minimum threshold (0.65).
- ❌ Unresolved dispute count (5) exceeds maximum allowed (0).
- ❌ 2 critical schema violation defects detected.

---
## 2. Annotator Reliability Leaderboard

| Annotator ID | Gold Evaluated | Gold Accuracy | Avg Confidence | Avg Latency | Reliability Tier |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `ann_lead_01` | `4` | `100.0%` | `0.94` | `8.4s` | **Tier 1 (Expert >= 90%)** |
| `ann_worker_02` | `4` | `100.0%` | `0.94` | `8.4s` | **Tier 1 (Expert >= 90%)** |
| `ann_worker_03` | `4` | `100.0%` | `0.94` | `8.4s` | **Tier 1 (Expert >= 90%)** |
| `ann_novice_04` | `4` | `50.0%` | `0.85` | `6.6s` | **Tier 4 (Flagged < 70%)** |

---

## 3. Detected Defect Log Sample

| Item ID | Annotator | Defect Type | Severity | Description |
| :--- | :--- | :--- | :---: | :--- |
| `defect_span_202` | `ann_novice_04` | `schema_invalid` | `CRITICAL` | Schema violation: 1 validation error for AnnotationItem |
| `defect_box_203` | `ann_novice_04` | `schema_invalid` | `CRITICAL` | Schema violation: 1 validation error for AnnotationItem |
| `gold_clf_001` | `ann_novice_04` | `speed_anomaly` | `MEDIUM` | Task completed in 1.2s (below 2.0s threshold). |
| `defect_oov_201` | `ann_novice_04` | `out_of_vocab` | `HIGH` | Label 'awesome_super_happy' is not in approved taxonomy. |

---

## 4. Provenance & Compliance Verification

- **Statistical Assumptions**: Krippendorff's Alpha computed across nominal classifications with chance correction.
- **Audit Immutability**: All flagged anomalies and expert resolutions are logged in version-controlled JSONL audit records.
- **Reproduction**: Run `annotatoriq audit --data data/sample/raw_multitask_annotations.jsonl` to re-execute.
