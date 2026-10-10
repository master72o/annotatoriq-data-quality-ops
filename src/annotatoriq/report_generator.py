"""
Report Generator Module for AnnotatorIQ.
Generates comprehensive Markdown and JSON Quality Operations Audit Reports.
"""

from typing import Dict, Any, List
from datetime import datetime, timezone
from annotatoriq.schema import QualityGateResult, DefectRecord


class ReportGenerator:
    """Generates standardized audit logs and human-readable QA summaries."""

    @staticmethod
    def generate_markdown_report(
        gate_result: QualityGateResult,
        annotator_stats: Dict[str, Any],
        defects: List[DefectRecord],
        dataset_name: str = "production_annotation_batch_v2",
    ) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        status_badge = "🟢 **PASSED ALL QUALITY GATES**" if gate_result.passed else "🔴 **FAILED QUALITY GATES**"

        md = f"""# 🛡️ AnnotatorIQ Data Quality Audit Report

**Dataset Batch:** `{dataset_name}`  
**Evaluation Timestamp:** `{timestamp}`  
**Gate Decision:** {status_badge}  

---

## 1. Executive Quality Summary

| Metric | Measured Value | Policy Threshold | Status |
| :--- | :---: | :---: | :---: |
| **Krippendorff's Alpha** | `{gate_result.krippendorff_alpha:.4f}` | `>= {gate_result.gate_thresholds['min_alpha']:.2f}` | {"✅ Pass" if gate_result.krippendorff_alpha >= gate_result.gate_thresholds['min_alpha'] else "❌ Fail"} |
| **Fleiss' Kappa** | `{gate_result.fleiss_kappa:.4f}` | `>= {gate_result.gate_thresholds['min_kappa']:.2f}` | {"✅ Pass" if gate_result.fleiss_kappa >= gate_result.gate_thresholds['min_kappa'] else "❌ Fail"} |
| **Gold Standard Accuracy** | `{gate_result.gold_accuracy * 100:.1f}%` | `>= {gate_result.gate_thresholds['min_gold_accuracy'] * 100:.1f}%` | {"✅ Pass" if gate_result.gold_accuracy >= gate_result.gate_thresholds['min_gold_accuracy'] else "❌ Fail"} |
| **Unresolved Disputes** | `{gate_result.unresolved_dispute_count}` (`{gate_result.unresolved_dispute_pct:.1f}%`) | `<= {gate_result.gate_thresholds['max_unresolved_disputes']}` | {"✅ Pass" if gate_result.unresolved_dispute_count <= gate_result.gate_thresholds['max_unresolved_disputes'] else "❌ Fail"} |
| **Total Defects Flagged** | `{gate_result.defect_count}` | — | {"⚠️ Review Required" if gate_result.defect_count > 0 else "✅ None"} |

"""
        if gate_result.failure_reasons:
            md += "### ⚠️ Active Release Blockers\n\n"
            for r in gate_result.failure_reasons:
                md += f"- ❌ {r}\n"
            md += "\n---\n"

        md += """## 2. Annotator Reliability Leaderboard

| Annotator ID | Gold Evaluated | Gold Accuracy | Avg Confidence | Avg Latency | Reliability Tier |
| :--- | :---: | :---: | :---: | :---: | :--- |
"""
        for a_id, s in annotator_stats.items():
            md += f"| `{a_id}` | `{s['gold_items_evaluated']}` | `{s['gold_accuracy_pct']}%` | `{s['avg_confidence']:.2f}` | `{s['avg_duration_sec']:.1f}s` | **{s['reliability_tier']}** |\n"

        md += "\n---\n\n## 3. Detected Defect Log Sample\n\n"
        if not defects:
            md += "*No defects detected in this dataset run.*\n"
        else:
            md += "| Item ID | Annotator | Defect Type | Severity | Description |\n"
            md += "| :--- | :--- | :--- | :---: | :--- |\n"
            for d in defects[:15]:
                md += f"| `{d.item_id}` | `{d.annotator_id}` | `{d.defect_type.value}` | `{d.severity}` | {d.description} |\n"
            if len(defects) > 15:
                md += f"\n*... and {len(defects) - 15} more defects logged in audit_trail.jsonl.*\n"

        md += """
---

## 4. Provenance & Compliance Verification

- **Statistical Assumptions**: Krippendorff's Alpha computed across nominal classifications with chance correction.
- **Audit Immutability**: All flagged anomalies and expert resolutions are logged in version-controlled JSONL audit records.
- **Reproduction**: Run `annotatoriq audit --data data/sample/raw_multitask_annotations.jsonl` to re-execute.
"""
        return md
