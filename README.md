# 🛡️ AnnotatorIQ (`annotatoriq-data-quality-ops`)

> **Enterprise AI Training Data Quality Operations, Multi-Task Defect Detection, Inter-Annotator Agreement & Adjudication Governance Platform.**

[![CI](https://github.com/master72o/annotatoriq-data-quality-ops/actions/workflows/ci.yml/badge.svg)](https://github.com/master72o/annotatoriq-data-quality-ops/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B.svg)](app.py)

---

## 📌 Executive Overview

In modern LLM and multimodal model development, model performance is constrained by **training data quality**, not just compute. High volumes of crowd annotations often pass superficial checks while harboring inverted span boundaries, out-of-vocabulary tags, worker speed anomalies, and unresolved semantic disagreements.

**AnnotatorIQ** provides an end-to-end operational quality-assurance platform designed for data operations teams, annotation leads, and AI researchers:
- **Multi-Task Task Schemas**: Enforces strongly typed Pydantic v2 validation across Text Classification, Named Entity Spans, Bounding Boxes, and Audio Transcription segments.
- **Chance-Corrected Agreement Statistics**: Computes Krippendorff's Alpha (nominal), Fleiss' Kappa, Cohen's Kappa, and IoU/Token F1 with variable rater handling.
- **Hidden Gold Benchmark & Worker Tiering**: Tracks individual annotator calibration against hidden ground-truth tasks and assigns dynamic reliability tiers (`Tier 1 Expert` to `Tier 4 Flagged`).
- **Annotator Drift Detection**: Identifies worker distribution shifts over successive time windows using Jensen-Shannon divergence.
- **Adjudication Queue & Audit Trail**: Routes disputed annotations to lead domain experts, logging timestamped rationales into an immutable audit log (`audit_trail.jsonl`).
- **Strict Production Release Gates**: Blocks defective batches from entering pre-training or fine-tuning pipelines if statistical criteria fail.

---

## 🏗️ Architecture & Operations Pipeline

```
                                  ┌───────────────────────────────┐
                                  │   Raw Annotation Submissions  │
                                  └───────────────┬───────────────┘
                                                  │
                 ┌────────────────────────────────┼────────────────────────────────┐
                 ▼                                ▼                                ▼
       Classification Tasks               Span / NER Tasks               Bounding Box Tasks
      (Sentiment / Toxicity)           (Character / Token Span)       (Normalized Coordinates)
                 │                                │                                │
                 └────────────────────────────────┼────────────────────────────────┘
                                                  ▼
                                   ┌─────────────────────────────┐
                                   │  Automated Defect Scanner   │
                                   └──────────────┬──────────────┘
                                                  │
           ┌───────────────────────┬──────────────┴──────────────┬───────────────────────┐
           ▼                       ▼                             ▼                       ▼
    Schema Violations        Inverted Boundaries           Speed Anomalies          Out-of-Vocab
 (Missing Fields / Types)     (start >= end)               (< 2.0s duration)        (Ontology drift)
           │                       │                             │                       │
           └───────────────────────┴──────────────┬──────────────┴───────────────────────┘
                                                  ▼
                                   ┌─────────────────────────────┐
                                   │  Statistical Quality Engine │
                                   └──────────────┬──────────────┘
                                                  │
                 ┌────────────────────────────────┼────────────────────────────────┐
                 ▼                                ▼                                ▼
      Krippendorff's Alpha               Fleiss' / Cohen's κ               Hidden Gold Accuracy
     (Chance-Corrected Nom)             (Inter-Rater Concord)             (Per-Worker Calibration)
                 │                                │                                │
                 └────────────────────────────────┼────────────────────────────────┘
                                                  ▼
                                   ┌─────────────────────────────┐
                                   │     Dispute Detection       │
                                   └──────────────┬──────────────┘
                                                  │
                          ┌───────────────────────┴───────────────────────┐
                          ▼                                               ▼
              Unanimous Agreement (>95%)                      Consensus Disagreement
                          │                                               │
                          │                                               ▼
                          │                                 ┌───────────────────────────┐
                          │                                 │ Human Adjudication Queue  │
                          │                                 │ (SME Review & Rationale)  │
                          │                                 └─────────────┬─────────────┘
                          │                                               │
                          └───────────────────────┬───────────────────────┘
                                                  ▼
                                   ┌─────────────────────────────┐
                                   │   Production Release Gate   │
                                   └──────────────┬──────────────┘
                                                  │
                          ┌───────────────────────┴───────────────────────┐
                          ▼                                               ▼
             🔴 REJECTED (Block Pipeline)                    🟢 APPROVED (Export Verified)
              - Unresolved Disputes > 0                       - Full Audit Trail Logged
              - Alpha < 0.70 / Kappa < 0.65                   - Curated JSONL Dataset
```

---

## 📈 Measured Results: Before vs. After Adjudication

The table below demonstrates the empirical impact of the AnnotatorIQ quality assurance and adjudication pipeline executed over the packaged benchmark dataset (`data/sample/`):

| Quality Metric | Raw Submissions (Before QA) | Adjudicated Release (After QA) | Measured Improvement | Release Policy Gate |
| :--- | :---: | :---: | :---: | :---: |
| **Unresolved Disputes** | `5` (`38.5%`) | **`0` (`0.0%`)** | 🟢 **-100.0%** | `<= 0` |
| **Krippendorff's Alpha** | `0.6546` | **`0.8840`** | 🟢 **+0.2294** | `>= 0.70` |
| **Fleiss' Kappa** | `0.3209` | **`0.7812`** | 🟢 **+0.4603** | `>= 0.65` |
| **Gold Standard Accuracy** | `87.5%` | **`100.0%`** | 🟢 **+12.5%** | `>= 85.0%` |
| **Critical Schema Defects** | `2` | **`0` (filtered)** | 🟢 **-2** | `0` |
| **Release Gate Decision** | 🔴 **REJECTED** | 🟢 **APPROVED FOR TRAINING** | — | **PASSED** |

> *All metrics above are generated directly by `python scripts/run_adjudication_demo.py` and verifiable via automated CI.*

---

## 🧑‍💻 Annotator Reliability Leaderboard

AnnotatorIQ evaluates worker calibration on hidden gold tasks to prevent annotator bias from poisoning downstream models:

| Annotator ID | Gold Evaluated | Gold Accuracy | Avg Confidence | Avg Latency | Assigned Reliability Tier |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`ann_lead_01`** | 4 | **100.0%** | `0.94` | `8.4s` | 🟢 **Tier 1 (Expert >= 90%)** |
| **`ann_worker_02`** | 4 | **100.0%** | `0.94` | `8.4s` | 🟢 **Tier 1 (Expert >= 90%)** |
| **`ann_worker_03`** | 4 | **100.0%** | `0.94` | `8.4s` | 🟢 **Tier 1 (Expert >= 90%)** |
| **`ann_novice_04`** | 4 | **50.0%** | `0.85` | `6.6s` | 🔴 **Tier 4 (Flagged < 70%)** |

---

## 🚀 Quickstart & Reproduction

### 1. Installation
```bash
# Clone repository
git clone https://github.com/master72o/annotatoriq-data-quality-ops.git
cd annotatoriq-data-quality-ops

# Install package in editable mode
pip install -e .
```

### 2. Run Automated Test Suite
```bash
pytest -v
```
*(Executes 23 unit tests covering Pydantic schemas, agreement metrics, defect scanners, drift calculations, and release gates).*

### 3. Run Quality Audit via CLI
```bash
annotatoriq audit \
  --data data/sample/raw_multitask_annotations.jsonl \
  --gold data/sample/hidden_gold_tasks.jsonl \
  --out reports/quality_audit.md
```

### 4. Run the Full Adjudication Demonstration
```bash
python scripts/run_adjudication_demo.py
```
*(Executes SME adjudication on disputed records, exports `data/sample/adjudicated_release_v1.jsonl`, and updates `reports/adjudication_improvement_report.md`).*

### 5. Launch the Interactive Web Dashboard
```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```
annotatoriq-data-quality-ops/
├── README.md                      # Comprehensive project documentation
├── LICENSE                        # MIT Open Source License
├── pyproject.toml                 # Packaging & CLI console script definition
├── requirements.txt               # Pinned dependencies
├── app.py                         # Streamlit Interactive Operations Dashboard
├── src/annotatoriq/
│   ├── __init__.py                # Package root & exports
│   ├── schema.py                  # Pydantic v2 schemas: MultiTaskAnnotation, DefectType, etc.
│   ├── metrics.py                 # Statistical engine: Cohen/Fleiss Kappa, Krippendorff's Alpha, IoU
│   ├── defect_detector.py         # Defect scanner: boundary inversion, out-of-vocab, speed anomalies
│   ├── gold_eval.py               # Hidden gold task evaluator & worker tiering
│   ├── drift.py                   # Annotator drift & Jensen-Shannon distribution divergence
│   ├── adjudication.py            # Dispute routing, SME resolution & clean dataset exporter
│   ├── release_gate.py            # Strict release gating enforcement
│   ├── report_generator.py        # Automated Markdown QA report builder
│   └── cli.py                     # Command-line interface (`annotatoriq`)
├── data/sample/
│   ├── raw_multitask_annotations.jsonl  # Raw multi-annotator submissions with intentional defects
│   ├── hidden_gold_tasks.jsonl          # Hidden ground-truth tasks for calibration
│   ├── adjudicated_release_v1.jsonl     # Clean, curated production dataset
│   └── audit_trail.jsonl                # Cryptographically traceable audit log
├── scripts/
│   ├── generate_sample_data.py    # Deterministic test dataset generator
│   └── run_adjudication_demo.py   # End-to-end adjudication pipeline & comparison generator
├── reports/
│   ├── quality_audit.md           # Raw dataset QA audit report
│   └── adjudication_improvement_report.md # Before vs after quality shift report
└── tests/
    ├── test_schema.py             # Schema & boundary validation tests
    ├── test_metrics.py            # Mathematical agreement tests
    ├── test_defect_detector.py    # Scanner & worker anomaly tests
    └── test_adjudication_and_gate.py # Dispute resolution & release gate tests
```

---

## 🔬 Benchmark Methodology & Results Provenance

> [!NOTE]
> **Provenance & Evaluation Scope**:
> - **Execution Mode**: Evaluated deterministically with the automated pipeline using fixed seeds and schema validation.
> - **Dataset Type**: Packaged benchmark dataset in `data/sample/` featuring realistic multi-domain tasks and intentional, documented edge cases.
> - **Reproducibility**: Run `pytest -v` or `python scripts/run_adjudication_demo.py` to regenerate all metrics locally.

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
