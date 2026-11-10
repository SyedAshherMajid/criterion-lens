"""Generate a privacy-conscious, provisional analysis of evaluation errors."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_RUNS = ("ft_1p5b", "ft_0p5b", "base_1p5b", "base_0p5b")
CATEGORY_PRIORITY = (
    "missing_information_guess",
    "unsupported_decision",
    "decision_polarity_confusion",
    "unnecessary_abstention",
    "applicability_confusion",
    "missed_gold_evidence",
    "invalid_sentence_id",
    "evidence_set_discrepancy",
    "label_mismatch",
)
DECISIVE_LABELS = {
    "included",
    "not included",
    "excluded",
    "not excluded",
}


def _categories(record: dict[str, Any]) -> list[str]:
    gold = record["gold"]
    gold_label = gold["label"]
    predicted_label = record["label"]
    gold_evidence = set(gold["evidence_ids"])
    predicted_evidence = set(record["evidence_ids"])
    categories: list[str] = []

    if predicted_label != gold_label:
        if gold_label == "not enough information" and predicted_label in DECISIVE_LABELS:
            categories.append("missing_information_guess")
        elif predicted_label == "not enough information" and gold_label in DECISIVE_LABELS:
            categories.append("unnecessary_abstention")
        elif "not applicable" in (gold_label, predicted_label):
            categories.append("applicability_confusion")
        elif predicted_label in DECISIVE_LABELS and gold_label in DECISIVE_LABELS:
            categories.append("decision_polarity_confusion")
        else:
            categories.append("label_mismatch")
    if record.get("unsupported_decision"):
        categories.append("unsupported_decision")
    if gold_evidence and not gold_evidence.intersection(predicted_evidence):
        categories.append("missed_gold_evidence")
    if predicted_evidence - gold_evidence:
        categories.append("evidence_set_discrepancy")
    if record.get("invalid_ids_dropped"):
        categories.append("invalid_sentence_id")
    return categories


def _diagnosis(categories: list[str]) -> str:
    diagnoses = {
        "missing_information_guess": (
            "The expert label is not-enough-information while the model made a decisive "
            "judgment; inspect whether the note actually supports that judgment."
        ),
        "unsupported_decision": (
            "The output marks a decisive judgment without cited evidence; this should remain "
            "a manual-review case."
        ),
        "decision_polarity_confusion": (
            "The predicted and expert labels are both decisive but disagree; verify polarity, "
            "negation, and criterion type against the note."
        ),
        "unnecessary_abstention": (
            "The expert supplied a decisive label while the model abstained; check whether "
            "the relevant fact is explicit in the note."
        ),
        "applicability_confusion": (
            "The disagreement involves not-applicable; verify whether the criterion can apply "
            "to the patient described."
        ),
        "missed_gold_evidence": (
            "The cited evidence does not overlap the expert evidence; inspect the sentence "
            "numbering and whether the model missed the supporting fact."
        ),
        "evidence_set_discrepancy": (
            "The model cited sentence IDs outside the expert evidence set; assess relevance "
            "locally because expert evidence may not be exhaustive."
        ),
        "invalid_sentence_id": (
            "The output contained one or more sentence IDs rejected by the parser; inspect "
            "the raw output locally."
        ),
        "label_mismatch": (
            "The model label differs from the expert label; compare the criterion and note "
            "locally before assigning a semantic cause."
        ),
    }
    return " ".join(diagnoses[category] for category in categories)


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation predictions: {path}")
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not records:
        raise ValueError(f"Evaluation predictions are empty: {path}")
    for record in records:
        if not isinstance(record.get("gold"), dict):
            raise ValueError(f"Prediction is missing its gold record: {path}")
    return records


def analyze_predictions(
    runs: dict[str, list[dict[str, Any]]],
    *,
    max_cases: int = 12,
) -> dict[str, Any]:
    if max_cases <= 0:
        raise ValueError("max_cases must be positive")
    candidates: dict[int, dict[str, Any]] = {}
    category_counts: Counter[str] = Counter()
    for model_name in MODEL_RUNS:
        for record in runs.get(model_name, []):
            categories = _categories(record)
            if not categories:
                continue
            category_counts.update(categories)
            annotation_id = int(record["annotation_id"])
            case = candidates.setdefault(
                annotation_id,
                {
                    "annotation_id": annotation_id,
                    "criterion_type": record["criterion_type"],
                    "criterion_text": record["criterion_text"][:240],
                    "gold_label": record["gold"]["label"],
                    "gold_evidence_ids": record["gold"]["evidence_ids"],
                    "model_outputs": [],
                },
            )
            case["model_outputs"].append(
                {
                    "model": model_name,
                    "predicted_label": record["label"],
                    "predicted_evidence_ids": record["evidence_ids"],
                    "categories": categories,
                    "provisional_diagnosis": _diagnosis(categories),
                }
            )

    ordered = sorted(
        candidates.values(),
        key=lambda case: (
            min(
                CATEGORY_PRIORITY.index(category)
                for output in case["model_outputs"]
                for category in output["categories"]
            ),
            case["annotation_id"],
        ),
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for category in CATEGORY_PRIORITY:
        if len(selected) >= max_cases:
            break
        for case in ordered:
            case_categories = {
                item for output in case["model_outputs"] for item in output["categories"]
            }
            if category in case_categories and case["annotation_id"] not in selected_ids:
                selected.append(case)
                selected_ids.add(case["annotation_id"])
                break
    for case in ordered:
        if len(selected) >= max_cases:
            break
        if case["annotation_id"] not in selected_ids:
            selected.append(case)
            selected_ids.add(case["annotation_id"])
    return {
        "schema_version": 1,
        "category_counts_by_model_record": dict(sorted(category_counts.items())),
        "unique_candidate_annotations": len(candidates),
        "selected_case_count": len(selected[:max_cases]),
        "cases": selected[:max_cases],
    }


def _markdown(result: dict[str, Any], split: str) -> str:
    lines = [
        f"# Provisional error analysis ({split} split)",
        "",
        (
            "These are automated candidates for author review, not clinical conclusions. "
            "The tracked report omits patient-note text; inspect the local prediction records "
            "before correcting any diagnosis. Evidence-set disagreement does not prove that "
            "a citation is clinically irrelevant, and contradictions require human review."
        ),
        "",
        f"Unique candidate annotations: {result['unique_candidate_annotations']}.",
        "",
        (
            "| Annotation | Type | Gold -> prediction | Gold evidence IDs | "
            "Predicted evidence IDs | Signals | Criterion |"
        ),
        "|---:|---|---|---|---|---|---|",
    ]
    for case in result["cases"]:
        for output in case["model_outputs"]:
            criterion = case["criterion_text"].replace("|", r"\|").replace("\n", " ")
            signals = ", ".join(output["categories"])
            lines.append(
                f"| {case['annotation_id']} ({output['model']}) "
                f"| {case['criterion_type']} "
                f"| {case['gold_label']} -> {output['predicted_label']} "
                f"| {case['gold_evidence_ids']} "
                f"| {output['predicted_evidence_ids']} "
                f"| {signals} | {criterion} |"
            )
            lines.append(f"|  |  |  |  |  | Diagnosis | {output['provisional_diagnosis']} |")
    lines.extend(
        [
            "",
            "## Candidate-category counts",
            "",
            "| Category | Model-record count |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| {category} | {count} |"
        for category, count in result["category_counts_by_model_record"].items()
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()
    runs = {
        run_name: _read_predictions(
            PROJECT_ROOT / "results" / "eval" / f"{args.split}_{run_name}" / "predictions.jsonl"
        )
        for run_name in MODEL_RUNS
    }
    result = analyze_predictions(runs, max_cases=args.max_cases)
    if args.split == "test" and result["unique_candidate_annotations"] < 10:
        raise RuntimeError(
            "Fewer than ten distinct test error candidates were found; "
            "do not duplicate cases to satisfy the report requirement"
        )
    output_json = args.output_json or (PROJECT_ROOT / "results" / "eval" / "error_cases.json")
    output_markdown = args.output_markdown or PROJECT_ROOT / "docs" / "ERROR_ANALYSIS.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output_markdown.write_text(_markdown(result, args.split), encoding="utf-8")
    print(f"Wrote {result['selected_case_count']} unique candidates for {args.split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
