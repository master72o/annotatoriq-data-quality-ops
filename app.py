"""
AnnotatorIQ: Interactive Training Data Quality Operations & Governance Dashboard.
Compatible with Streamlit Community Cloud and Hugging Face Spaces.
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

from annotatoriq.schema import (
    AnnotationItem,
    GoldItem,
    TaskType,
    DefectRecord,
    AdjudicationRecord,
)
from annotatoriq.defect_detector import DefectDetector
from annotatoriq.gold_eval import GoldEvaluator
from annotatoriq.release_gate import QualityGate
from annotatoriq.adjudication import AdjudicationEngine
from annotatoriq.metrics import AgreementMetrics
from annotatoriq.drift import DriftDetector
from annotatoriq.report_generator import ReportGenerator

st.set_page_config(
    page_title="AnnotatorIQ — AI Data Quality Operations",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e2530;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #2d3748;
        margin-bottom: 12px;
    }
    .badge-pass {
        background-color: #064e3b;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-fail {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_sample_dataset() -> tuple[List[AnnotationItem], List[GoldItem], List[DefectRecord]]:
    data_path = Path("data/sample/raw_multitask_annotations.jsonl")
    gold_path = Path("data/sample/hidden_gold_tasks.jsonl")

    annotations = []
    schema_defects = []
    if data_path.exists():
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    annotations.append(AnnotationItem.model_validate_json(line_str))
                except Exception as e:
                    try:
                        raw = json.loads(line_str)
                        schema_defects.append(
                            DefectRecord(
                                item_id=raw.get("item_id", "unknown"),
                                annotator_id=raw.get("annotator_id", "unknown"),
                                defect_type=DefectType.SCHEMA_INVALID,
                                description=f"Schema violation: {str(e).splitlines()[0]}",
                                severity="CRITICAL",
                                observed_value=line_str[:120],
                            )
                        )
                    except Exception:
                        pass

    gold_items = []
    if gold_path.exists():
        with open(gold_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    gold_items.append(GoldItem.model_validate_json(line.strip()))

    return annotations, gold_items, schema_defects


annotations, gold_items, schema_defects = load_sample_dataset()

# Initialize session state for adjudications
if "adjudications" not in st.session_state:
    st.session_state.adjudications = []

# Sidebar Controls
st.sidebar.title("🛡️ AnnotatorIQ")
st.sidebar.caption("AI Training Data Quality & Governance Engine v2.0")
st.sidebar.markdown("---")
st.sidebar.subheader("Quality Gate Settings")
min_alpha = st.sidebar.slider("Min Krippendorff's Alpha", 0.50, 0.95, 0.70, 0.05)
min_kappa = st.sidebar.slider("Min Fleiss' Kappa", 0.50, 0.95, 0.65, 0.05)
min_gold = st.sidebar.slider("Min Gold Accuracy (%)", 70, 95, 85, 5) / 100.0

st.sidebar.markdown("---")
st.sidebar.info("""
**Audited & Verified**:
- Multi-Task Quality Assurance
- Chance-Corrected Agreement
- Hidden Gold Tracking
- Immutable Audit Trail
""")

# Run analytics
detector = DefectDetector()
scanned_defects = detector.scan_annotations(annotations)
defects = schema_defects + scanned_defects
gold_eval = GoldEvaluator.evaluate_annotators(annotations, gold_items) if gold_items else {}
fatigue_profiles = DriftDetector.compute_annotator_fatigue_index(annotations)

gate = QualityGate(
    min_krippendorff_alpha=min_alpha,
    min_fleiss_kappa=min_kappa,
    min_gold_accuracy=min_gold,
)
gate_res = gate.evaluate_dataset(
    annotations,
    gold_items=gold_items,
    adjudications=st.session_state.adjudications,
)

disputes = AdjudicationEngine.identify_disputes(annotations)
resolved_ids = {a.item_id for a in st.session_state.adjudications}
unresolved_disputes = [d for d in disputes if d["item_id"] not in resolved_ids]

# Header
st.title("🛡️ AnnotatorIQ — Data Quality Operations Platform")
st.markdown("Automated pre-training data quality assurance, worker calibration, defect detection, and adjudication queue.")

# Top Metrics Row
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total Records", len(annotations))
with c2:
    st.metric("Krippendorff's α", f"{gate_res.krippendorff_alpha:.3f}")
with c3:
    st.metric("Fleiss' κ", f"{gate_res.fleiss_kappa:.3f}")
with c4:
    st.metric("Avg Gold Accuracy", f"{gate_res.gold_accuracy * 100:.1f}%")
with c5:
    status_label = "✅ PASSED" if gate_res.passed else "❌ GATED"
    st.metric("Release Gate", status_label)

st.markdown("---")

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset Health & Defect Scanner",
    "🏆 Annotator Leaderboard & Drift",
    "⚖️ Adjudication Queue",
    "📦 Production Release & Audit Log",
])

# TAB 1: DEFECT SCANNER
with tab1:
    st.subheader("Automated Pre-Training Defect Detection")
    st.write("Scans raw annotation submissions for schema violations, inverted boundaries, out-of-vocab labels, and speed anomalies.")

    if not defects:
        st.success("No defects detected in current dataset batch.")
    else:
        d_df = pd.DataFrame([
            {
                "Item ID": d.item_id,
                "Annotator": d.annotator_id,
                "Defect Type": d.defect_type.value,
                "Severity": d.severity,
                "Description": d.description,
            }
            for d in defects
        ])
        st.dataframe(d_df, use_container_width=True)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("#### Defect Distribution by Type")
            st.bar_chart(d_df["Defect Type"].value_counts())
        with colB:
            st.markdown("#### Severity Breakdown")
            st.bar_chart(d_df["Severity"].value_counts())

# TAB 2: LEADERBOARD & DRIFT
with tab2:
    st.subheader("Annotator Calibration & Reliability Leaderboard")
    st.write("Annotators are ranked by objective performance against hidden gold tasks, confidence calibration, and latency.")

    if gold_eval:
        l_df = pd.DataFrame([
            {
                "Annotator ID": a_id,
                "Gold Accuracy (%)": data["gold_accuracy_pct"],
                "Gold Tasks Evaluated": data["gold_items_evaluated"],
                "Avg Confidence": data["avg_confidence"],
                "Avg Duration (s)": data["avg_duration_sec"],
                "Reliability Tier": data["reliability_tier"],
            }
            for a_id, data in gold_eval.items()
        ]).sort_values(by="Gold Accuracy (%)", ascending=False)
        st.dataframe(l_df, use_container_width=True)
    else:
        st.info("No gold standard evaluation records available.")

    st.markdown("---")
    st.subheader("Worker Fatigue & Entropy Drift Telemetry")
    st.write("Audits pacing degradation, repetitive labeling patterns, and rapid velocity spikes indicating annotator burnout or botting.")

    if fatigue_profiles:
        f_df = pd.DataFrame([
            {
                "Annotator ID": w_id,
                "Fatigue Score": prof.get("fatigue_score", 0.0),
                "Speed Drop": f"{prof.get('speed_drop_ratio', 0.0) * 100:.1f}%",
                "1st Half Avg Sec": f"{prof.get('first_half_avg_sec', 0.0):.1f}s",
                "2nd Half Avg Sec": f"{prof.get('second_half_avg_sec', 0.0):.1f}s",
                "Drift / Alert Status": prof.get("status", "NORMAL"),
            }
            for w_id, prof in fatigue_profiles.items()
        ])
        st.dataframe(f_df, use_container_width=True)
    else:
        st.info("No worker sequence history available for fatigue indexing.")

# TAB 3: ADJUDICATION QUEUE
with tab3:
    st.subheader(f"Interactive Adjudication Queue ({len(unresolved_disputes)} Unresolved)")
    st.write("When annotators disagree on subjective or complex items, cases are routed here for expert review.")

    if not unresolved_disputes:
        st.success("🎉 All annotation disputes have been successfully adjudicated!")
    else:
        selected_dispute = st.selectbox(
            "Select Disputed Item to Review:",
            options=unresolved_disputes,
            format_func=lambda x: f"{x['item_id']} ({x['task_type'].value}) — Votes: {x['vote_distribution'] if 'vote_distribution' in x else 'Disagreement'}",
        )

        if selected_dispute:
            st.markdown(f"**Item ID:** `{selected_dispute['item_id']}`")
            st.markdown(f"**Prompt / Text:** *\"{selected_dispute.get('input_text', 'N/A')}\"*")

            st.write("#### Conflicting Submissions:")
            for ann in selected_dispute["annotations"]:
                st.write(f"- Annotator **{ann.get('annotator_id')}**: Label `{ann.get('label')}` (Confidence: {ann.get('confidence', 1.0)})")

            st.write("#### Expert Adjudication Decision:")
            with st.form(key=f"adj_form_{selected_dispute['item_id']}"):
                final_label = st.selectbox(
                    "Final Ground Truth Label:",
                    options=["positive", "negative", "neutral", "ambiguous", "toxic", "safe"],
                )
                rationale = st.text_area(
                    "Adjudication Rationale (logged to immutable audit log):",
                    placeholder="Explain why this decision was chosen according to annotation guidelines...",
                )
                submit_adj = st.form_submit_button("Submit Adjudication & Update Ground Truth")

                if submit_adj:
                    adj_rec = AdjudicationEngine.resolve_dispute(
                        item_id=selected_dispute["item_id"],
                        task_type=selected_dispute["task_type"],
                        disputed_options=selected_dispute["annotations"],
                        final_decision={"label": final_label},
                        adjudicator_id="lead_sme_adjudicator",
                        rationale=rationale or "Adjudicated per standard guideline definitions.",
                    )
                    st.session_state.adjudications.append(adj_rec)
                    st.success(f"Item `{selected_dispute['item_id']}` resolved and appended to audit log!")
                    st.rerun()

# TAB 4: PRODUCTION RELEASE GATE & AUDIT
with tab4:
    st.subheader("Release Gating Decision & Audit Trail")

    if gate_res.passed:
        st.success("🟢 **DATASET APPROVED FOR MODEL TRAINING**: All quality gates and agreement criteria satisfied.")
    else:
        st.error("🔴 **DATASET REJECTED BY RELEASE GATE**: The following quality standards failed:")
        for r in gate_res.failure_reasons:
            st.write(f"- ❌ {r}")

    st.markdown("---")
    st.subheader("Export Clean Dataset & Audit Trail")
    colX, colY = st.columns(2)
    with colX:
        clean_data = AdjudicationEngine.export_adjudicated_dataset(annotations, st.session_state.adjudications)
        clean_json = json.dumps(clean_data, indent=2)
        st.download_button(
            "📥 Download Verified Dataset (JSON)",
            data=clean_json,
            file_name="adjudicated_release_v1.json",
            mime="application/json",
        )
    with colY:
        audit_records = [a.model_dump() for a in st.session_state.adjudications]
        audit_json = json.dumps(audit_records, indent=2)
        st.download_button(
            "📜 Download Audit Trail Log (JSON)",
            data=audit_json,
            file_name="audit_trail.json",
            mime="application/json",
        )

    st.markdown("---")
    st.subheader("Formal Quality Operations Audit Report")
    report_md = ReportGenerator.generate_markdown_report(
        gate_res,
        gold_eval,
        defects,
        fatigue_profiles=fatigue_profiles,
        dataset_name="sample_production_batch",
    )
    with st.expander("📄 View Standardized Markdown Audit Report", expanded=False):
        st.markdown(report_md)

    st.download_button(
        "📋 Download Formal Audit Report (Markdown)",
        data=report_md,
        file_name="annotatoriq_audit_report.md",
        mime="text/markdown",
    )
