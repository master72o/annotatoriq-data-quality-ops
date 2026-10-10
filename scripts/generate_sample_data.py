"""
Benchmark dataset generator for AnnotatorIQ.
Creates realistic multi-task annotation records with intentional edge cases, defects, and gold standards.
"""

import json
from pathlib import Path

sample_dir = Path("data/sample")
sample_dir.mkdir(parents=True, exist_ok=True)

# 1. Hidden Gold Tasks
gold_items = [
    {
        "item_id": "gold_clf_001",
        "task_type": "classification",
        "expected_label": "toxic",
        "domain": "safety",
        "rationale": "Direct personal insult and abusive language."
    },
    {
        "item_id": "gold_clf_002",
        "task_type": "classification",
        "expected_label": "positive",
        "domain": "customer_support",
        "rationale": "Unambiguous customer gratitude and satisfaction."
    },
    {
        "item_id": "gold_clf_003",
        "task_type": "classification",
        "expected_label": "neutral",
        "domain": "technical",
        "rationale": "Factual statement describing hardware specification."
    },
    {
        "item_id": "gold_clf_004",
        "task_type": "classification",
        "expected_label": "safe",
        "domain": "safety",
        "rationale": "Benign informational query about encryption."
    },
    {
        "item_id": "gold_ner_001",
        "task_type": "span_ner",
        "expected_spans": [
            {"label": "PER", "start": 0, "end": 14, "text": "Albert Einstein"},
            {"label": "DATE", "start": 25, "end": 29, "text": "1879"}
        ],
        "domain": "biography",
        "rationale": "Standard historical entities."
    },
    {
        "item_id": "gold_box_001",
        "task_type": "bounding_box",
        "expected_boxes": [
            {"label": "document_table", "xmin": 0.10, "ymin": 0.20, "xmax": 0.90, "ymax": 0.60}
        ],
        "domain": "finance",
        "rationale": "Full quarterly balance sheet table coordinates."
    }
]

with open(sample_dir / "hidden_gold_tasks.jsonl", "w", encoding="utf-8") as f:
    for g in gold_items:
        f.write(json.dumps(g) + "\n")

# 2. Raw Annotations (Multi-Annotator submissions)
raw_annotations = []
annotators = ["ann_lead_01", "ann_worker_02", "ann_worker_03", "ann_novice_04"]

# A. Gold items annotated by all 4
for ann in annotators:
    # gold_clf_001 (all agree toxic except novice who misses it)
    raw_annotations.append({
        "item_id": "gold_clf_001",
        "annotator_id": ann,
        "task_type": "classification",
        "input_text": "You are completely incompetent and should be fired.",
        "label": "toxic" if ann != "ann_novice_04" else "neutral",
        "duration_seconds": 8.5 if ann != "ann_novice_04" else 1.2,  # Speed anomaly on novice
        "confidence": 0.95 if ann != "ann_novice_04" else 0.60,
    })
    # gold_clf_002 (all agree positive)
    raw_annotations.append({
        "item_id": "gold_clf_002",
        "annotator_id": ann,
        "task_type": "classification",
        "input_text": "Thank you so much! That resolved my issue immediately.",
        "label": "positive",
        "duration_seconds": 6.2,
        "confidence": 0.98,
    })
    # gold_clf_003 (all agree neutral)
    raw_annotations.append({
        "item_id": "gold_clf_003",
        "annotator_id": ann,
        "task_type": "classification",
        "input_text": "The server has 64GB of ECC DDR5 RAM installed.",
        "label": "neutral",
        "duration_seconds": 7.1,
        "confidence": 0.92,
    })
    # gold_clf_004 (novice fails again)
    raw_annotations.append({
        "item_id": "gold_clf_004",
        "annotator_id": ann,
        "task_type": "classification",
        "input_text": "How does RSA-2048 public key encryption work?",
        "label": "safe" if ann != "ann_novice_04" else "ambiguous",
        "duration_seconds": 12.0,
        "confidence": 0.90,
    })

# B. Regular Production Classification Items (varying consensus & genuine ambiguity)
prod_items = [
    ("task_clf_101", "The UI update looks clean, but performance took a slight hit.", "ambiguous", "neutral", "negative"),
    ("task_clf_102", "Fatal error encountered during system initialization.", "negative", "negative", "negative"),
    ("task_clf_103", "Customer requested cancellation within 30-day window.", "neutral", "neutral", "neutral"),
    ("task_clf_104", "Great work on the turnaround time, excellent service!", "positive", "positive", "positive"),
    ("task_clf_105", "I am not sure if this is supposed to happen.", "ambiguous", "ambiguous", "neutral"),
    ("task_clf_106", "Your software crashed and erased my unsaved file.", "negative", "negative", "negative"),
    ("task_clf_107", "Monthly active users increased by 14% quarter over quarter.", "neutral", "positive", "neutral"),
    ("task_clf_108", "Please close my ticket, I figured it out myself.", "neutral", "neutral", "neutral"),
]

for item_id, text, l1, l2, l3 in prod_items:
    raw_annotations.append({"item_id": item_id, "annotator_id": "ann_lead_01", "task_type": "classification", "input_text": text, "label": l1, "duration_seconds": 9.0, "confidence": 0.90})
    raw_annotations.append({"item_id": item_id, "annotator_id": "ann_worker_02", "task_type": "classification", "input_text": text, "label": l2, "duration_seconds": 8.0, "confidence": 0.85})
    raw_annotations.append({"item_id": item_id, "annotator_id": "ann_worker_03", "task_type": "classification", "input_text": text, "label": l3, "duration_seconds": 11.0, "confidence": 0.88})

# C. Seeded Intentional Defects (to prove DefectDetector catches them!)
# Defect 1: Out-of-vocabulary label
raw_annotations.append({
    "item_id": "defect_oov_201",
    "annotator_id": "ann_novice_04",
    "task_type": "classification",
    "input_text": "The transaction went through without any issues.",
    "label": "awesome_super_happy",  # OOV!
    "duration_seconds": 5.0,
    "confidence": 0.70,
})

# Defect 2: Inverted span boundaries
raw_annotations.append({
    "item_id": "defect_span_202",
    "annotator_id": "ann_novice_04",
    "task_type": "span_ner",
    "input_text": "Patient was diagnosed with Hypertension in London.",
    "spans": [{"label": "LOC", "start": 48, "end": 42, "text": "London"}],  # start > end! Inverted!
    "duration_seconds": 4.0,
    "confidence": 0.50,
})

# Defect 3: Out-of-bounds coordinates
raw_annotations.append({
    "item_id": "defect_box_203",
    "annotator_id": "ann_novice_04",
    "task_type": "bounding_box",
    "input_uri": "s3://docs/invoice_001.pdf",
    "boxes": [{"label": "document_table", "xmin": 0.20, "ymin": 0.30, "xmax": 1.45, "ymax": 0.80}],  # xmax > 1.0!
    "duration_seconds": 3.0,
    "confidence": 0.60,
})

with open(sample_dir / "raw_multitask_annotations.jsonl", "w", encoding="utf-8") as f:
    for a in raw_annotations:
        f.write(json.dumps(a) + "\n")

print(f"Generated {len(raw_annotations)} raw annotations and {len(gold_items)} gold standard tasks.")
