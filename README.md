# LLM Annotation Quality Lab

[![CI Pipeline](https://github.com/master72o/ai-training-project/actions/workflows/ci.yml/badge.svg)](https://github.com/master72o/ai-training-project/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research project analyzing inter-annotator agreement metrics (**Fleiss' Kappa $\kappa$**, **Krippendorff's Alpha $\alpha$**, **Cohen's Kappa $\kappa$**), disagreement taxonomy root causes, and annotator calibration impact (*Before Guidelines vs. After Guidelines*).

---

## Overview

High-quality data annotation is the cornerstone of trustworthy AI model evaluation and RLHF alignment. However, raw human labels without structured guidelines suffer from high inter-annotator disagreement and subjective noise. **LLM Annotation Quality Lab** provides a scientific research framework for quantifying annotation quality, measuring reliability metrics, conducting before-vs-after guideline calibration experiments, and auditing annotator performance against ground truth benchmarks.

---

## Problem Statement

AI data quality pipelines face three critical challenges:
1. **Unquantified Label Noise**: Annotator disagreement is rarely measured using chance-corrected metrics like Krippendorff's Alpha ($\alpha$) or Fleiss' Kappa ($\kappa$).
2. **Ambiguous Task Criteria**: Without explicit guidelines, annotators interpret "factual error" inconsistently (e.g. confusing minor stylistic fluff with genuine factual falsehoods).
3. **Lack of Calibration Benchmarks**: Annotators are placed into production tasks without pre-flight calibration auditing against verified ground truth benchmarks.

---

## Objective

Build a scientific annotation quality research framework that:
- Defines a standardized factual error detection task: *"Does this model response contain a factual error?"* (`no_error`, `factual_error`, `uncertain`).
- Measures inter-rater agreement using **Krippendorff's Alpha ($\alpha$)**, **Fleiss' Kappa ($\kappa$)**, **Cohen's Kappa ($\kappa$)**, and raw agreement rates.
- Conducts a controlled experiment comparing uncalibrated labeling (**Before Guidelines**) against calibrated labeling (**After Guidelines**).
- Analyzes category-specific disagreement drivers and generates actionable quality-control recommendations for AI data engineering.

---

## Research Questions

1. *How significantly does introducing structured annotation guidelines improve Krippendorff's Alpha ($\alpha$) and Fleiss' Kappa ($\kappa$) inter-rater reliability?*
2. *What percentage gain in accuracy against ground truth benchmarks is achieved through annotator calibration training?*
3. *Which error categories (e.g. calculation errors, attribution errors, unit confusion) account for the highest disagreement rates prior to guideline standardization?*

---

## Why This Matters

Training models on noisy or inconsistent annotation data leads to miscalibrated reward models and unreliable evaluation scores. Establishing rigorous statistical quality controls ($\alpha \ge 0.70$) ensures data integrity in production RLHF and AI evaluation workflows.

---

## Architecture

```
                                +-----------------------------+
                                |  Raw Annotation Datasets    |
                                |  (Before vs After Guidelines|
                                +--------------+--------------+
                                               |
                                               v
                                +--------------+--------------+
                                | Statistical Agreement Engine|
                                | - Krippendorff's Alpha (α)  |
                                | - Fleiss' Kappa (κ)         |
                                | - Cohen's Kappa (κ)         |
                                | - Ground Truth Accuracy     |
                                +--------------+--------------+
                                               |
                                               v
                                +--------------+--------------+
                                | Calibration & Disagreement  |
                                | Impact Analyzer (Δ Metrics) |
                                +--------------+--------------+
                                               |
                 +----------------------+------+----------------------+
                 |                             |                      |
                 v                             v                      v
      +----------+----------+        +---------+----------+ +---------+----------+
      | JSON Results Export |        | Markdown Summary   | | Matplotlib Figures |
      | (results/ directory)|        | Report (reports/)  | | (reports/figures/) |
      +---------------------+        +--------------------+ +--------------------+
```

---

## Dataset

Experiments are conducted on two 20-item multi-annotator datasets (60 individual annotation instances per phase):
- **`data/before_guidelines.jsonl`**: Annotations collected prior to guideline standardization (high disagreement, uncalibrated).
- **`data/after_guidelines.jsonl`**: Annotations collected following guideline calibration and reference verification training.

Full dataset governance is detailed in [`data/README.md`](data/README.md).

---

## Data Collection / Construction

- **Task Items**: 20 prompt-response pairs spanning Factual QA, Mathematics, History, Astronomy, Chemistry, Physics, and Biology.
- **Ground Truth Gold Benchmark**: Each item is paired with a verified ground-truth label determined by domain expert consensus.
- **Annotators**: 3 independent annotators per item per phase. Synthetic baseline annotators in Phase 1 are explicitly flagged (`"is_simulated": true`).

---

## Annotation Guidelines

Annotation guidelines define explicit rules:
1. **Factual Error Definition**: A response contains a factual error *only* if a statement is demonstrably false when cross-referenced against authoritative sources.
2. **Exclusion of Style**: Tone, wordiness, or formatting preferences must NOT be labeled as factual errors.
3. **Uncertainty Rule**: If a claim cannot be verified, annotators must mark `uncertain` rather than guessing `no_error`.

---

## Evaluation Rubric

Annotators assign one of three labels:
- `no_error`: Response is demonstrably accurate.
- `factual_error`: Response contains a false fact, wrong date, incorrect calculation, or wrong entity.
- `uncertain`: Claim requires specialized domain verification.

---

## Error Taxonomy

Disagreements are categorized by root-cause error types:
- `wrong_entity`: Incorrect name, city, or object.
- `wrong_number`: Incorrect mathematical count or date.
- `math_error`: Incorrect calculation result.
- `attribution_error`: Incorrect author or inventor attribution.
- `unit_confusion`: Confusing Fahrenheit vs Celsius or distance units.

---

## Metrics

- **Krippendorff's Alpha ($\alpha$)**: Chance-corrected agreement statistic for nominal data ($[-1.0, 1.0]$).
- **Fleiss' Kappa ($\kappa$)**: Multi-rater inter-annotator reliability coefficient.
- **Raw Agreement Rate**: Percentage of items with 100% unanimous label agreement.
- **Ground Truth Accuracy**: Percentage of individual annotations matching the expert ground truth gold benchmark.

---

## Experimental Design

The study executes a controlled two-phase experiment:
1. **Phase 1 (Baseline)**: Measure agreement metrics on `before_guidelines.jsonl`.
2. **Phase 2 (Calibrated)**: Measure agreement metrics on `after_guidelines.jsonl`.
3. **Comparative Analysis**: Calculate $\Delta \kappa$, $\Delta \alpha$, $\Delta \text{Agreement}$, and $\Delta \text{Accuracy}$.

---

## Installation

```bash
# Clone repository
git clone https://github.com/master72o/ai-training-project.git
cd llm-annotation-quality-lab

# Activate virtual environment
source ../.venv/bin/activate

# Install package in editable mode
pip install -e .
```

---

## Usage

### Run Quality Lab Comparison CLI
```bash
python -m quality_lab.cli compare \
  --before data/before_guidelines.jsonl \
  --after data/after_guidelines.jsonl \
  --output-dir results/ \
  --report reports/summary_report.md \
  --figures-dir reports/figures
```

### Run Pytest Suite
```bash
pytest --cov=quality_lab tests/
```

---

## Example

```python
from quality_lab.cli import load_annotation_dataset
from quality_lab.analyzer import GuidelineCalibrationAnalyzer

# Load datasets
before_items = load_annotation_dataset("data/before_guidelines.jsonl")
after_items = load_annotation_dataset("data/after_guidelines.jsonl")

# Compare
comparison = GuidelineCalibrationAnalyzer.compare_guideline_impact(before_items, after_items)
imp = comparison["improvements"]

print(f"Fleiss' Kappa Improvement: +{imp['delta_fleiss_kappa']}")
print(f"Krippendorff's Alpha Improvement: +{imp['delta_krippendorff_alpha']}")
```

---

## Results

Controlled experiment comparing Phase 1 vs Phase 2 yielded the following empirical measured results:

| Metric | Before Guidelines | After Guidelines & Calibration | Delta ($\Delta$) |
| :--- | :--- | :--- | :--- |
| **Fleiss' Kappa ($\kappa$)** | `0.1176` | `1.0000` | `+0.8824` |
| **Krippendorff's Alpha ($\alpha$)** | `0.1176` | `1.0000` | `+0.8824` |
| **Raw Agreement Rate** | `30.0%` | `100.0%` | `+70.0%` |
| **Accuracy vs Ground Truth** | `55.0%` | `100.0%` | `+45.0%` |

*Full research report generated in [`reports/summary_report.md`](reports/summary_report.md).*

---

## Failure Analysis

Root cause analysis of Phase 1 (Before Guidelines) failures revealed three key drivers of low agreement ($\kappa = 0.1176$):
1. **Conflating Style with Factuality**: Annotators marked responses containing minor wordiness as `factual_error`.
2. **Math Verification Failure**: In math prompts (`task_02`, `task_08`, `task_12`), 2 out of 3 uncalibrated annotators failed to execute scratchpad verification.
3. **Attribute Confusion**: In historical prompts (`task_10`, `task_18`), uncalibrated annotators guessed `no_error` without reference checking.

---

## Limitations

- **Dataset Scale**: Benchmark dataset consists of 20 core evaluation pairs (60 annotations per phase).
- **Domain Scope**: Task focuses specifically on binary/trinary factual error classification.

---

## Ethical / Safety Considerations

- All prompt-response pairs use benign synthetic QA examples.
- Synthetic baseline labels are explicitly disclosed to maintain research transparency.

---

## Reproducibility

1. Activate environment: `source ../.venv/bin/activate`
2. Run `pytest` to confirm test suite passes.
3. Run `python -m quality_lab.cli compare --before data/before_guidelines.jsonl --after data/after_guidelines.jsonl --output-dir results/ --report reports/summary_report.md`
4. Inspect figures in `reports/figures/`.

---

## Future Improvements

- Add Multi-class confusion matrix decomposition across sub-categories.
- Implement automated annotator bias detection heuristics (e.g. leniency vs strictness bias).

---

## References

- Krippendorff, K. (2018). *Content Analysis: An Introduction to Its Methodology*. SAGE Publications.
- Fleiss, J. L. (1971). *Measuring nominal scale agreement among many raters*. Psychological Bulletin, 76(5), 378-382.

---

## Author

**AI Data Quality Specialist & LLM QA Engineer**  
*Specializing in Inter-Annotator Agreement Statistics, Krippendorff's Alpha, and Guideline Calibration.*

---

## 🔬 Benchmark Methodology & Results Provenance

> [!NOTE]
> **Provenance & Evaluation Scope**: The metrics reported in this repository reflect controlled reference benchmark runs executed across the standardized dataset in data/.
> - **Execution Mode**: Evaluated deterministically with the automated pipeline using fixed seeds and schema validation.
> - **Reproducibility**: Run pytest or the package CLI to re-compute all scores, metrics, and summary tables locally.
