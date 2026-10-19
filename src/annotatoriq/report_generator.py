"""
Report Generator Module for AnnotatorIQ.
Generates comprehensive Markdown and JSON Quality Operations Audit Reports.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from annotatoriq.schema import QualityGateResult, DefectRecord


class ReportGenerator:
    """Generates standardized audit logs, machine-readable telemetry, and human-readable QA summaries."""

    @staticmethod
    def generate_markdown_report(
        gate_result: QualityGateResult,
        annotator_stats: Dict[str, Any],
        defects: List[DefectRecord],
        fatigue_profiles: Optional[Dict[str, Any]] = None,
        dataset_name: str = "production_annotation_batch_v2",
    ) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        status_badge = "🟢 **PASSED ALL QUALITY GATES**" if gate_result.passed else "🔴 **FAILED QUALITY GATES**"

        max_drift = gate_result.gate_thresholds.get("max_drift_defects", 2)
        drift_status = "✅ Pass" if gate_result.defect_count <= max_drift else "⚠️ Flagged"

        md = f"""# 🛡️ AnnotatorIQ Data Quality Audit Report

**Dataset Batch:** `{dataset_name}`  
**Evaluation Timestamp:** `{timestamp}`  
**Gate Decision:** {status_badge}  

---

## 1. Executive Quality Summary

| Metric | Measured Value | Policy Threshold | Status |
| :--- | :---: | :---: | :---: |
| **Krippendorff's Alpha** | `{gate_result.krippendorff_alpha:.4f}` | `>= {gate_result.gate_thresholds.get('min_alpha', 0.70):.2f}` | {"✅ Pass" if gate_result.krippendorff_alpha >= gate_result.gate_thresholds.get('min_alpha', 0.70) else "❌ Fail"} |
| **Fleiss' Kappa** | `{gate_result.fleiss_kappa:.4f}` | `>= {gate_result.gate_thresholds.get('min_kappa', 0.65):.2f}` | {"✅ Pass" if gate_result.fleiss_kappa >= gate_result.gate_thresholds.get('min_kappa', 0.65) else "❌ Fail"} |
| **Gold Standard Accuracy** | `{gate_result.gold_accuracy * 100:.1f}%` | `>= {gate_result.gate_thresholds.get('min_gold_accuracy', 0.85) * 100:.1f}%` | {"✅ Pass" if gate_result.gold_accuracy >= gate_result.gate_thresholds.get('min_gold_accuracy', 0.85) else "❌ Fail"} |
| **Unresolved Disputes** | `{gate_result.unresolved_dispute_count}` (`{gate_result.unresolved_dispute_pct:.1f}%`) | `<= {gate_result.gate_thresholds.get('max_unresolved_disputes', 0)}` | {"✅ Pass" if gate_result.unresolved_dispute_count <= gate_result.gate_thresholds.get('max_unresolved_disputes', 0) else "❌ Fail"} |
| **Total Defects Flagged** | `{gate_result.defect_count}` | `<= {max_drift} (drift allowance)` | {drift_status} |

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
            md += f"| `{a_id}` | `{s.get('gold_items_evaluated', 0)}` | `{s.get('gold_accuracy_pct', 0.0)}%` | `{s.get('avg_confidence', 0.0):.2f}` | `{s.get('avg_duration_sec', 0.0):.1f}s` | **{s.get('reliability_tier', 'N/A')}** |\n"

        if fatigue_profiles:
            md += "\n---\n\n## 3. Worker Fatigue & Entropy Drift Telemetry\n\n"
            md += "| Annotator ID | Fatigue Score | Speed Drop Ratio | Drift / Alert Status |\n"
            md += "| :--- | :---: | :---: | :--- |\n"
            for w_id, prof in fatigue_profiles.items():
                f_score = prof.get("fatigue_score", 0.0)
                speed_drop = prof.get("speed_drop_ratio", 0.0)
                status = prof.get("status", "NORMAL")
                status_icon = "⚠️" if f_score > 0.50 else "🟢"
                md += f"| `{w_id}` | `{f_score:.2f}` | `{speed_drop * 100:.1f}%` | {status_icon} **{status}** |\n"

        md += "\n---\n\n## 4. Detected Defect Log Sample\n\n"
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

## 5. Provenance & Compliance Verification

- **Statistical Assumptions**: Krippendorff's Alpha computed across nominal classifications with chance correction.
- **Audit Immutability**: All flagged anomalies and expert resolutions are logged in version-controlled JSONL audit records.
- **Reproduction**: Run `annotatoriq audit --data data/sample/raw_multitask_annotations.jsonl` to re-execute.
"""
        return md

    @staticmethod
    def generate_json_report(
        gate_result: QualityGateResult,
        annotator_stats: Dict[str, Any],
        defects: List[DefectRecord],
        fatigue_profiles: Optional[Dict[str, Any]] = None,
        dataset_name: str = "production_annotation_batch_v2",
    ) -> Dict[str, Any]:
        """Generates a structured, machine-readable quality audit report dictionary for CI/CD gates."""
        timestamp = datetime.now(timezone.utc).isoformat()
        return {
            "dataset_name": dataset_name,
            "timestamp": timestamp,
            "gate_decision": "PASSED" if gate_result.passed else "FAILED",
            "passed": gate_result.passed,
            "metrics": {
                "krippendorff_alpha": gate_result.krippendorff_alpha,
                "fleiss_kappa": gate_result.fleiss_kappa,
                "gold_accuracy": gate_result.gold_accuracy,
                "unresolved_dispute_count": gate_result.unresolved_dispute_count,
                "unresolved_dispute_pct": gate_result.unresolved_dispute_pct,
                "total_defects_flagged": gate_result.defect_count,
            },
            "gate_thresholds": gate_result.gate_thresholds,
            "active_blockers": gate_result.failure_reasons,
            "annotator_stats": annotator_stats,
            "fatigue_telemetry": fatigue_profiles or {},
            "defects_sample": [
                {
                    "item_id": d.item_id,
                    "annotator_id": d.annotator_id,
                    "defect_type": d.defect_type.value,
                    "severity": d.severity,
                    "description": d.description,
                }
                for d in defects[:25]
            ],
            "total_defects_count": len(defects),
        }
