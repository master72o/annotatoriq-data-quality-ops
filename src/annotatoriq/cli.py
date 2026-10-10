"""
Command Line Interface (CLI) for AnnotatorIQ.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from annotatoriq.schema import AnnotationItem, GoldItem, AdjudicationRecord, DefectRecord, DefectType
from annotatoriq.defect_detector import DefectDetector
from annotatoriq.gold_eval import GoldEvaluator
from annotatoriq.release_gate import QualityGate
from annotatoriq.report_generator import ReportGenerator
from annotatoriq.adjudication import AdjudicationEngine


def load_annotations(path: str) -> tuple[List[AnnotationItem], List[DefectRecord]]:
    items = []
    defects = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                items.append(AnnotationItem.model_validate_json(line_str))
            except Exception as e:
                try:
                    raw = json.loads(line_str)
                    defects.append(
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
    return items, defects


def load_gold(path: str) -> List[GoldItem]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(GoldItem.model_validate_json(line.strip()))
    return items


def main():
    parser = argparse.ArgumentParser(description="AnnotatorIQ Quality Operations CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # audit command
    audit_parser = subparsers.add_parser("audit", help="Run full QA audit on dataset")
    audit_parser.add_argument("--data", required=True, help="Path to annotations JSONL")
    audit_parser.add_argument("--gold", required=False, help="Path to hidden gold items JSONL")
    audit_parser.add_argument("--out", required=False, default="reports/quality_audit.md", help="Output report path")

    # gate command
    gate_parser = subparsers.add_parser("gate", help="Evaluate production release gate")
    gate_parser.add_argument("--data", required=True, help="Path to annotations JSONL")
    gate_parser.add_argument("--gold", required=False, help="Path to hidden gold items JSONL")
    gate_parser.add_argument("--min-alpha", type=float, default=0.70)
    gate_parser.add_argument("--min-kappa", type=float, default=0.65)
    gate_parser.add_argument("--min-gold", type=float, default=0.85)

    args = parser.parse_args()

    if args.command == "audit":
        annotations, schema_defects = load_annotations(args.data)
        gold_items = load_gold(args.gold) if args.gold else []

        detector = DefectDetector()
        detected_defects = detector.scan_annotations(annotations)
        all_defects = schema_defects + detected_defects

        gold_stats = GoldEvaluator.evaluate_annotators(annotations, gold_items) if gold_items else {}

        gate = QualityGate()
        gate_res = gate.evaluate_dataset(annotations, gold_items=gold_items)
        # Add schema defects count to gate result if any
        if schema_defects:
            gate_res.defect_count += len(schema_defects)
            gate_res.passed = False
            gate_res.failure_reasons.append(f"{len(schema_defects)} critical schema violation defects detected.")

        report = ReportGenerator.generate_markdown_report(gate_res, gold_stats, all_defects)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")

        print(f"[AnnotatorIQ] Audit complete. Report written to {out_path}")
        print(f"[AnnotatorIQ] Gate Passed: {gate_res.passed} | Defects Found: {len(all_defects)} | Alpha: {gate_res.krippendorff_alpha}")

    elif args.command == "gate":
        annotations, schema_defects = load_annotations(args.data)
        gold_items = load_gold(args.gold) if args.gold else []
        gate = QualityGate(
            min_krippendorff_alpha=args.min_alpha,
            min_fleiss_kappa=args.min_kappa,
            min_gold_accuracy=args.min_gold,
        )
        res = gate.evaluate_dataset(annotations, gold_items=gold_items)
        if schema_defects:
            res.defect_count += len(schema_defects)
            res.passed = False
            res.failure_reasons.append(f"{len(schema_defects)} critical schema violation defects detected.")
        print(json.dumps(res.dict(), indent=2))
        sys.exit(0 if res.passed else 1)


if __name__ == "__main__":
    main()
