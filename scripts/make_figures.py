"""Generate report figures and LaTeX tables from committed result files."""

import argparse
import csv
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml

from criterion_lens.eval.metrics import (
    bootstrap_confidence_intervals,
    classification_metrics,
)

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ("base_1p5b", "ft_1p5b", "base_0p5b", "ft_0p5b")
LABEL_SHORT = {
    "included": "Incl.",
    "not included": "Not incl.",
    "not enough information": "NEI",
    "not applicable": "N/A",
    "excluded": "Excl.",
    "not excluded": "Not excl.",
}


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _load_evaluations(results_root: Path) -> dict[str, dict[str, Any]]:
    evaluations: dict[str, dict[str, Any]] = {}
    for path in sorted((results_root / "eval").glob("*/metrics.json")):
        run_name = path.parent.name
        if not (run_name.startswith("val_") or run_name.startswith("test_")):
            continue
        metrics = _read_json(path)
        if metrics.get("split") != run_name.split("_", maxsplit=1)[0]:
            raise ValueError(f"Split metadata does not match evaluation directory: {path}")
        if not metrics.get("bootstrap_95_ci"):
            predictions_path = path.parent / "predictions.jsonl"
            if not predictions_path.is_file():
                raise FileNotFoundError(
                    f"Predictions needed for bootstrap confidence intervals: {predictions_path}"
                )
            predictions = [
                json.loads(line)
                for line in predictions_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            gold = [row["gold"]["label"] for row in predictions]
            predicted = [row["label"] for row in predictions]
            criterion_types = [row["criterion_type"] for row in predictions]
            intervals = bootstrap_confidence_intervals(gold, predicted)
            observed = metrics["classification"]["overall"]
            recomputed = classification_metrics(gold, predicted, criterion_types)["overall"]
            accuracy_matches = np.isclose(
                observed["accuracy"], recomputed["accuracy"], atol=1e-12
            )
            f1_matches = np.isclose(
                observed["fixed_label_macro_f1"],
                recomputed["fixed_label_macro_f1"],
                atol=1e-12,
            )
            if not accuracy_matches or not f1_matches:
                raise ValueError(
                    f"Prediction metrics do not match saved metrics: {predictions_path}"
                )
            metrics["bootstrap_95_ci"] = intervals
        if run_name in evaluations:
            raise ValueError(f"Duplicate evaluation result: {run_name}")
        evaluations[run_name] = metrics
    if not evaluations:
        raise FileNotFoundError(
            f"No validation or test metrics found under {results_root / 'eval'}"
        )
    return evaluations


def _save_figure(figure: plt.Figure, name: str, *destinations: Path) -> None:
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination / name, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _ordered_names(names: list[str]) -> list[str]:
    order = {name: index for index, name in enumerate(MODEL_ORDER)}
    return sorted(names, key=lambda name: order.get(name.split("_", 1)[1], len(order)))


def _make_evaluation_figures(
    evaluations: dict[str, dict[str, Any]],
    figures_dir: Path,
    report_figures_dir: Path,
) -> None:
    names = sorted(
        evaluations,
        key=lambda name: (
            0 if name.startswith("val_") else 1,
            MODEL_ORDER.index(name.split("_", 1)[1])
            if name.split("_", 1)[1] in MODEL_ORDER
            else len(MODEL_ORDER),
        ),
    )
    labels = [_display_model(name) for name in names]
    accuracy = [evaluations[name]["classification"]["overall"]["accuracy"] for name in names]
    macro_f1 = [
        evaluations[name]["classification"]["overall"]["fixed_label_macro_f1"] for name in names
    ]
    accuracy_intervals = [
        evaluations[name].get("bootstrap_95_ci", {}).get("accuracy") for name in names
    ]
    macro_f1_intervals = [
        evaluations[name].get("bootstrap_95_ci", {}).get("macro_f1") for name in names
    ]
    accuracy_errors = np.asarray(
        [
            [
                max(0.0, score - interval[0]),
                max(0.0, interval[1] - score),
            ]
            if interval
            else [0.0, 0.0]
            for score, interval in zip(accuracy, accuracy_intervals, strict=True)
        ]
    ).T
    macro_f1_errors = np.asarray(
        [
            [
                max(0.0, score - interval[0]),
                max(0.0, interval[1] - score),
            ]
            if interval
            else [0.0, 0.0]
            for score, interval in zip(macro_f1, macro_f1_intervals, strict=True)
        ]
    ).T

    positions = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(max(8, len(names) * 1.2), 4.8))
    width = 0.36
    axis.bar(
        positions - width / 2,
        accuracy,
        width,
        label="Accuracy",
        yerr=accuracy_errors,
        capsize=3,
    )
    axis.bar(
        positions + width / 2,
        macro_f1,
        width,
        label="Fixed-label macro-F1",
        yerr=macro_f1_errors,
        capsize=3,
    )
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Criterion-level classification (95% bootstrap CIs)")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save_figure(
        figure,
        "classification_metrics.png",
        figures_dir,
        report_figures_dir,
    )

    labels_order = evaluations[names[0]]["classification"]["overall"]["labels"]
    figure, axis = plt.subplots(figsize=(max(9, len(labels_order) * 1.3), 5))
    width = 0.8 / len(names)
    for index, name in enumerate(names):
        per_class = evaluations[name]["classification"]["overall"]["per_class"]
        scores = [per_class[label]["f1-score"] for label in labels_order]
        offsets = np.arange(len(labels_order)) - 0.4 + width / 2 + index * width
        axis.bar(offsets, scores, width, label=labels[index])
    axis.set_xticks(np.arange(len(labels_order)), [LABEL_SHORT[label] for label in labels_order])
    axis.set_ylim(0, 1)
    axis.set_ylabel("F1")
    axis.set_title("Per-class fixed-label F1")
    axis.legend(fontsize="small", ncol=2)
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, "per_class_f1.png", figures_dir, report_figures_dir)

    evidence_micro = [evaluations[name]["evidence"]["overall"]["micro"]["f1"] for name in names]
    evidence_macro = [
        evaluations[name]["evidence"]["overall"]["macro_nonempty_gold"]["f1"] for name in names
    ]
    figure, axis = plt.subplots(figsize=(max(8, len(names) * 1.2), 4.8))
    axis.bar(positions - width / 2, evidence_micro, width, label="Micro F1")
    axis.bar(positions + width / 2, evidence_macro, width, label="Macro F1 (non-empty gold)")
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("F1")
    axis.set_title("Evidence-set extraction")
    axis.legend(fontsize="small")
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, "evidence_f1.png", figures_dir, report_figures_dir)

    for name in names:
        source_dir = figures_dir.parent / "eval" / name
        for image in source_dir.glob("confusion_*.png"):
            shutil.copy2(image, figures_dir / f"{name}_{image.name}")
            shutil.copy2(image, report_figures_dir / f"{name}_{image.name}")


def _make_training_figure(results_root: Path, figures_dir: Path, report_figures_dir: Path) -> None:
    histories = sorted((results_root / "train").glob("**/log_history.json"))
    if not histories:
        LOGGER.info("Skipping training curves: no log_history.json files found")
        return
    figure, (loss_axis, eval_axis) = plt.subplots(1, 2, figsize=(11, 4.5))
    plotted = False
    for path in histories:
        rows = json.loads(path.read_text(encoding="utf-8"))
        name = path.parent.name.replace("qwen25_", "").replace("_lora", "").replace("_", " ")
        train_rows = [row for row in rows if "loss" in row and "step" in row]
        eval_rows = [row for row in rows if "eval_loss" in row and "epoch" in row]
        if train_rows:
            loss_axis.plot(
                [row["step"] for row in train_rows],
                [row["loss"] for row in train_rows],
                marker=".",
                label=name,
            )
            plotted = True
        if eval_rows:
            eval_axis.plot(
                [row["epoch"] for row in eval_rows],
                [row["eval_loss"] for row in eval_rows],
                marker="o",
                label=name,
            )
            plotted = True
    if not plotted:
        plt.close(figure)
        LOGGER.info("Skipping training curves: log histories have no loss data")
        return
    loss_axis.set_title("Training loss")
    loss_axis.set_xlabel("Optimizer step")
    loss_axis.set_ylabel("Loss")
    loss_axis.grid(alpha=0.25)
    eval_axis.set_title("Validation loss (not generation F1)")
    eval_axis.set_xlabel("Epoch")
    eval_axis.set_ylabel("Loss")
    eval_axis.grid(alpha=0.25)
    loss_axis.legend()
    eval_axis.legend()
    _save_figure(figure, "training_curves.png", figures_dir, report_figures_dir)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _number(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    return float(value)


def _format_optional_score(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "--"


def _display_model(run_name: str) -> str:
    variant, size = run_name.split("_", maxsplit=1)[1].split("_", maxsplit=1)
    variant_label = {"base": "Base", "ft": "Fine-tuned"}.get(variant, variant.title())
    return f"{variant_label} {size.replace('p', '.', 1).upper()}"


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def _write_table(
    path: Path,
    caption: str,
    headers: list[str],
    rows: list[list[str]],
    *,
    label: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    alignment = "l" + "r" * (len(headers) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
    ]
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.extend(
        [
            rf"\begin{{tabular}}{{{alignment}}}",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
        ]
    )
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_evaluation_tables(evaluations: dict[str, dict[str, Any]], tables_dir: Path) -> None:
    names = sorted(evaluations)
    result_rows = []
    evidence_rows = []
    for name in names:
        metrics = evaluations[name]
        model = _display_model(name)
        overall = metrics["classification"]["overall"]
        ci = metrics.get("bootstrap_95_ci", {})
        accuracy_ci = ci.get("accuracy")
        f1_ci = ci.get("macro_f1")
        result_rows.append(
            [
                _latex_escape(metrics["split"].upper()),
                _latex_escape(model),
                f"{overall['accuracy']:.3f}",
                f"{overall['fixed_label_macro_f1']:.3f}",
                (f"[{accuracy_ci[0]:.3f}, {accuracy_ci[1]:.3f}]" if accuracy_ci else "--"),
                f"[{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]" if f1_ci else "--",
            ]
        )
        evidence = metrics["evidence"]["overall"]
        evidence_rows.append(
            [
                _latex_escape(metrics["split"].upper()),
                _latex_escape(model),
                f"{evidence['micro']['f1']:.3f}",
                f"{evidence['macro_nonempty_gold']['f1']:.3f}",
                str(evidence["empty_gold"]["n_records"]),
                _format_optional_score(evidence["empty_gold"]["abstention_accuracy"]),
            ]
        )
    _write_table(
        tables_dir / "main_results.tex",
        "Criterion-level bootstrap 95\\% confidence intervals (1,000 resamples; seed 42).",
        ["Split", "Model", "Accuracy", "Macro-F1", "Accuracy CI", "Macro-F1 CI"],
        result_rows,
        label="tab:main-results",
    )
    _write_table(
        tables_dir / "evidence_results.tex",
        "Evidence-set scores; primary macro-F1 averages records with non-empty expert evidence.",
        ["Split", "Model", "Micro-F1", "Primary macro-F1", "Empty gold", "Abstention acc."],
        evidence_rows,
        label="tab:evidence-results",
    )

    preferred = "test_ft_1p5b" if "test_ft_1p5b" in evaluations else "val_ft_1p5b"
    per_class = evaluations[preferred]["classification"]["overall"]["per_class"]
    labels = evaluations[preferred]["classification"]["overall"]["labels"]
    _write_table(
        tables_dir / "per_class_results.tex",
        f"Per-class precision, recall, and F1 for {preferred.replace('_', ' ')}.",
        ["Label", "Precision", "Recall", "F1", "Support"],
        [
            [
                _latex_escape(label),
                f"{per_class[label]['precision']:.3f}",
                f"{per_class[label]['recall']:.3f}",
                f"{per_class[label]['f1-score']:.3f}",
                str(int(per_class[label]["support"])),
            ]
            for label in labels
        ],
        label="tab:per-class",
    )


def _write_data_table_and_macros(
    results_root: Path,
    evaluations: dict[str, dict[str, Any]],
    tables_dir: Path,
) -> None:
    audit = _read_json(results_root / "data_audit.json")
    manifest = _read_json(PROJECT_ROOT / "data" / "splits" / "split_manifest.json")
    split_rows = []
    for split_name in ("train", "val", "test"):
        split = manifest[split_name]
        split_rows.append(
            [
                split_name.upper(),
                str(split["n_patients"]),
                str(split["n_annotations"]),
                str(split["n_inclusion"]),
                str(split["n_exclusion"]),
            ]
        )
    _write_table(
        tables_dir / "split_statistics.tex",
        "Patient-disjoint dataset partition sizes.",
        ["Split", "Patients", "Annotations", "Inclusion", "Exclusion"],
        split_rows,
        label="tab:split",
    )

    macros = [
        rf"\newcommand{{\DatasetAnnotationCount}}{{{audit['n_annotations']}}}",
        rf"\newcommand{{\DatasetPatientCount}}{{{audit['n_patients']}}}",
        rf"\newcommand{{\DatasetTrialCount}}{{{audit['n_trials']}}}",
    ]
    for split_name in ("train", "val", "test"):
        split = manifest[split_name]
        key = split_name.title()
        macros.extend(
            [
                rf"\newcommand{{\{key}PatientCount}}{{{split['n_patients']}}}",
                rf"\newcommand{{\{key}AnnotationCount}}{{{split['n_annotations']}}}",
            ]
        )

    validation = evaluations.get("val_ft_1p5b")
    validation_f1 = (
        validation["classification"]["overall"]["fixed_label_macro_f1"] if validation else None
    )
    macros.append(
        rf"\newcommand{{\ValidationHeadlineMacroFone}}{{{validation_f1:.4f}}}"
        if validation_f1 is not None
        else r"\newcommand{\ValidationHeadlineMacroFone}{--}"
    )
    headline = evaluations.get("test_ft_1p5b")
    if headline:
        overall = headline["classification"]["overall"]
        intervals = headline.get("bootstrap_95_ci", {})

        def _interval(name: str) -> str:
            values = intervals.get(name)
            return f"[{values[0]:.3f}, {values[1]:.3f}]" if values else "--"

        macros.extend(
            [
                rf"\newcommand{{\TestHeadlineAccuracy}}{{{overall['accuracy']:.4f}}}",
                rf"\newcommand{{\TestHeadlineMacroFone}}{{{overall['fixed_label_macro_f1']:.4f}}}",
                rf"\newcommand{{\TestHeadlineAccuracyCI}}{{{_interval('accuracy')}}}",
                rf"\newcommand{{\TestHeadlineMacroFoneCI}}{{{_interval('macro_f1')}}}",
            ]
        )
    else:
        macros.extend(
            [
                r"\newcommand{\TestHeadlineAccuracy}{\textit{pending final evaluation}}",
                r"\newcommand{\TestHeadlineMacroFone}{\textit{pending final evaluation}}",
                r"\newcommand{\TestHeadlineAccuracyCI}{--}",
                r"\newcommand{\TestHeadlineMacroFoneCI}{--}",
            ]
        )
    (tables_dir / "result_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")


def _write_training_config_table(tables_dir: Path) -> None:
    rows = []
    for label, file_name in (
        ("1.5B", "train_qwen25_1p5b.yaml"),
        ("0.5B", "train_qwen25_0p5b.yaml"),
    ):
        config = yaml.safe_load(
            (PROJECT_ROOT / "configs" / file_name).read_text(encoding="utf-8")
        )
        lora = config["lora"]
        rows.append(
            [
                label,
                str(config["max_seq_len"]),
                str(config["num_train_epochs"]),
                (
                    f"{config['per_device_train_batch_size']} x "
                    f"{config['gradient_accumulation_steps']}"
                ),
                f"{lora['r']}/{lora['lora_alpha']}",
                f"{config['learning_rate']:.1e}",
                str(config["seed"]),
            ]
        )
    _write_table(
        tables_dir / "training_hyperparameters.tex",
        "LoRA training configuration read from the versioned YAML files.",
        ["Model", "Max tokens", "Epochs", "Batch x accum.", "Rank/alpha", "Learning rate", "Seed"],
        rows,
        label="tab:hyperparameters",
    )


def _write_error_table(results_root: Path, tables_dir: Path, *, required: bool) -> None:
    path = results_root / "eval" / "error_cases.json"
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Final error analysis is missing: {path}")
        LOGGER.info("Skipping error table: error_cases.json is unavailable")
        return
    errors = _read_json(path)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Provisional error candidates for author review; note text is omitted.}",
        r"\label{tab:error-cases}",
        r"\begin{tabularx}{\textwidth}{rllX}",
        r"\toprule",
        r"ID & Variant/type & Gold $\rightarrow$ predicted & Candidate analysis \\",
        r"\midrule",
    ]
    for case in errors["cases"]:
        if not case["model_outputs"]:
            continue
        output = case["model_outputs"][0]
        variant = _latex_escape(output["model"]) + "/" + _latex_escape(case["criterion_type"])
        labels = (
            _latex_escape(case["gold_label"])
            + " $\\rightarrow$ "
            + _latex_escape(output["predicted_label"])
        )
        categories = ", ".join(_latex_escape(item) for item in output["categories"])
        diagnosis = _latex_escape(output["provisional_diagnosis"])
        lines.append(
            f"{case['annotation_id']} & {variant} & {labels} & {categories}: {diagnosis} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table*}",
            "",
        ]
    )
    (tables_dir / "error_cases.tex").write_text("\n".join(lines), encoding="utf-8")


def _load_kv_files(bench_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    files = []
    for path in sorted(bench_dir.rglob("kv_cache.json")):
        data = _read_json(path)
        if data.get("input_source") != "deterministic_synthetic_token_ids":
            raise ValueError(f"KV benchmark was not run on synthetic tokens: {path}")
        files.append((path.parent.name, data))
    return files


def _make_kv_figures(
    files: list[tuple[str, dict[str, Any]]],
    figures_dir: Path,
    report_figures_dir: Path,
    tables_dir: Path,
) -> None:
    if not files:
        LOGGER.info("Skipping KV figures: kv_cache.json files are unavailable")
        return
    benchmark_rows = []
    plots: list[tuple[str, list[dict[str, Any]]]] = []
    for label, data in files:
        rows = data.get("benchmarks", [])
        if not rows:
            continue
        preferred_dtype = (
            "float16" if any(row["dtype"] == "float16" for row in rows) else rows[0]["dtype"]
        )
        max_output = max(row["output_length"] for row in rows)
        selected = [
            row
            for row in rows
            if row["dtype"] == preferred_dtype and row["output_length"] == max_output
        ]
        plots.append((label, selected))
        for row in selected:
            benchmark_rows.append(
                [
                    _latex_escape(label),
                    str(row["input_length"]),
                    str(row["output_length"]),
                    row["dtype"],
                    f"{row['total_latency_speedup']:.2f}"
                    if row["total_latency_speedup"] is not None
                    else "--",
                ]
            )

    for metric, file_name, title, ylabel in (
        ("total_latency_s", "kv_latency.png", "No-cache and KV-cache latency", "Latency (s)"),
        (
            "output_tokens_per_second",
            "kv_throughput.png",
            "KV-cache decode throughput",
            "Output tokens/s",
        ),
        ("peak_memory_bytes", "kv_memory.png", "Peak allocated GPU memory", "Bytes"),
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        for label, rows in plots:
            for path_name in ("no_cache", "kv_cache"):
                points = []
                for row in rows:
                    summary = row[path_name].get(metric)
                    value = summary.get("mean") if isinstance(summary, dict) else summary
                    if value is not None:
                        points.append((row["input_length"], float(value)))
                if points:
                    points.sort()
                    axis.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        marker="o",
                        label=f"{label} {path_name}",
                    )
        if not axis.lines:
            plt.close(figure)
            LOGGER.info("Skipping %s: benchmark contains no finite %s values", file_name, metric)
            continue
        axis.set_title(title)
        axis.set_xlabel("Input length (tokens)")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize="small")
        axis.grid(alpha=0.25)
        _save_figure(figure, file_name, figures_dir, report_figures_dir)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for label, rows in plots:
        points = sorted(
            (point["cache_len"], point["cache_bytes"])
            for row in rows
            for point in row.get("cache_growth", [])
        )
        if points:
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker=".",
                label=label,
            )
    if axis.lines:
        axis.set_title("Measured KV allocation versus cached sequence length")
        axis.set_xlabel("Cached key/value length (tokens)")
        axis.set_ylabel("Allocated cache bytes")
        axis.legend(fontsize="small")
        axis.grid(alpha=0.25)
        _save_figure(figure, "kv_cache_growth.png", figures_dir, report_figures_dir)
    else:
        plt.close(figure)
        LOGGER.info("Skipping KV cache growth plot: cache growth records are unavailable")

    _write_table(
        tables_dir / "kv_summary.tex",
        "KV-cache latency speedup on deterministic synthetic token IDs.",
        ["Run", "Input tokens", "Output tokens", "Dtype", "Speedup"],
        benchmark_rows,
        label="tab:kv-results",
    )


def _make_attention_figures(
    bench_dir: Path,
    figures_dir: Path,
    report_figures_dir: Path,
    tables_dir: Path,
) -> None:
    accuracy_files = sorted(bench_dir.rglob("attention_accuracy.csv"))
    perf_files = sorted(bench_dir.rglob("attention_perf.csv"))
    if not accuracy_files:
        LOGGER.info("Skipping attention figures: accuracy CSV is unavailable")
        return
    accuracy_rows = [row for path in accuracy_files for row in _load_csv(path)]
    worst_rows = sorted(
        (row for row in accuracy_rows if row["max_abs_error"] not in ("", "None")),
        key=lambda row: float(row["max_abs_error"]),
        reverse=True,
    )[:12]
    _write_table(
        tables_dir / "attention_accuracy.tex",
        "Largest observed absolute attention errors across the configured correctness grid.",
        ["Dtype", "Q length", "K length", "Head dim", "Block", "Max abs. error", "Within tol."],
        [
            [
                row["dtype"],
                row["query_length"],
                row["key_length"],
                row["head_dim"],
                row["block_q"],
                f"{float(row['max_abs_error']):.3g}",
                row["within_tolerance"],
            ]
            for row in worst_rows
        ],
        label="tab:attention-accuracy",
    )
    if worst_rows:
        figure, axis = plt.subplots(figsize=(10, max(4, len(worst_rows) * 0.34)))
        table_rows = [
            [
                row["dtype"],
                f"{row['query_length']}/{row['key_length']}",
                row["head_dim"],
                row["block_q"],
                f"{float(row['max_abs_error']):.2g}",
                row["within_tolerance"],
            ]
            for row in worst_rows
        ]
        axis.axis("off")
        axis.table(
            cellText=table_rows,
            colLabels=["Dtype", "Q/K", "Head dim", "Block", "Max abs err.", "Within tol."],
            loc="center",
        )
        axis.set_title("Largest tiled-attention numerical errors")
        _save_figure(figure, "attention_accuracy.png", figures_dir, report_figures_dir)

    if not perf_files:
        LOGGER.info("Skipping attention performance plot: performance CSV is unavailable")
        return
    performance_rows = [row for path in perf_files for row in _load_csv(path)]
    successful = [
        row
        for row in performance_rows
        if row["status"] == "OK"
        and row["dtype"] == "float16"
        and row["causal"].lower() == "false"
        and row["head_dim"] == "64"
    ]
    if not successful:
        LOGGER.info("Skipping attention performance plot: no matching successful CUDA rows")
        return
    figure, (time_axis, memory_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    for method in sorted({row["method"] for row in successful}):
        rows = sorted(
            (row for row in successful if row["method"] == method),
            key=lambda row: int(row["sequence_length"]),
        )
        time_axis.plot(
            [int(row["sequence_length"]) for row in rows],
            [float(row["mean_s"]) for row in rows],
            marker="o",
            label=method,
        )
        memory_points = [
            (int(row["sequence_length"]), _number(row["peak_memory_bytes"])) for row in rows
        ]
        memory_axis.plot(
            [point[0] for point in memory_points if point[1] is not None],
            [point[1] for point in memory_points if point[1] is not None],
            marker="o",
            label=method,
        )
    time_axis.set_title("Attention latency")
    time_axis.set_xlabel("Sequence length")
    time_axis.set_ylabel("Mean time (s)")
    memory_axis.set_title("Attention peak allocation")
    memory_axis.set_xlabel("Sequence length")
    memory_axis.set_ylabel("Peak memory (bytes)")
    for axis in (time_axis, memory_axis):
        axis.grid(alpha=0.25)
        axis.legend(fontsize="x-small")
    _save_figure(figure, "attention_performance.png", figures_dir, report_figures_dir)


def _make_speculative_figures(
    bench_dir: Path,
    figures_dir: Path,
    report_figures_dir: Path,
    tables_dir: Path,
) -> None:
    files = sorted(bench_dir.rglob("speculative.json"))
    if not files:
        LOGGER.info("Skipping speculative figures: speculative.json files are unavailable")
        return
    rows_by_group: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for path in files:
        data = _read_json(path)
        if data.get("input_source") != "synthetic":
            raise ValueError(f"Speculative benchmark used non-synthetic prompts: {path}")
        for row in data.get("results", []):
            key = (str(row["draft_model"]), int(row["max_new"]), int(row["k"]))
            rows_by_group[key].append(row)

    summaries = []
    for (draft, output_length, k), rows in sorted(rows_by_group.items()):
        acceptance = [row["acceptance_rate"] for row in rows if row["acceptance_rate"] is not None]
        speedups = [row["speedup"] for row in rows if row["speedup"] is not None]
        target_latencies = [row["target_latency_s"]["mean"] for row in rows]
        speculative_latencies = [row["speculative_latency_s"]["mean"] for row in rows]
        summaries.append(
            {
                "draft": draft,
                "max_new": output_length,
                "k": k,
                "acceptance_rate": float(np.mean(acceptance)) if acceptance else None,
                "speedup": float(np.mean(speedups)) if speedups else None,
                "target_latency_s": float(np.mean(target_latencies)),
                "speculative_latency_s": float(np.mean(speculative_latencies)),
            }
        )
    for metric, file_name, title, ylabel in (
        (
            "acceptance_rate",
            "speculative_acceptance.png",
            "Draft-token acceptance",
            "Acceptance rate",
        ),
        (
            "speedup",
            "speculative_speedup.png",
            "Speculative decoding speedup",
            "Target/speculative latency",
        ),
        (
            "target_latency_s",
            "speculative_latency_breakdown.png",
            "Target and speculative latency",
            "Mean latency (s)",
        ),
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        for draft, output_length in sorted({(row["draft"], row["max_new"]) for row in summaries}):
            metric_series = (
                ("target_latency_s", "target"),
                ("speculative_latency_s", "speculative"),
            ) if metric == "target_latency_s" else ((metric, None),)
            for series_metric, series_label in metric_series:
                points = sorted(
                    (
                        row["k"],
                        row[series_metric],
                    )
                    for row in summaries
                    if row["draft"] == draft
                    and row["max_new"] == output_length
                    and row[series_metric] is not None
                )
                if points:
                    suffix = f" {series_label}" if series_label else ""
                    axis.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        marker="o",
                        label=f"{draft}, output={output_length}{suffix}",
                    )
        if axis.lines:
            axis.set_title(title)
            axis.set_xlabel("Speculative lookahead k")
            axis.set_ylabel(ylabel)
            axis.legend(fontsize="small")
            axis.grid(alpha=0.25)
            _save_figure(figure, file_name, figures_dir, report_figures_dir)
        else:
            plt.close(figure)
            LOGGER.info("Skipping %s: no finite rows", file_name)

    _write_table(
        tables_dir / "speculative_summary.tex",
        "Speculative decoding means over synthetic prompts, grouped by output length.",
        ["Draft", "Output tokens", "k", "Acceptance", "Speedup"],
        [
            [
                _latex_escape(row["draft"]),
                str(row["max_new"]),
                str(row["k"]),
                f"{row['acceptance_rate']:.3f}" if row["acceptance_rate"] is not None else "--",
                f"{row['speedup']:.2f}" if row["speedup"] is not None else "--",
            ]
            for row in summaries
        ],
        label="tab:speculative",
    )


def generate(
    results_root: Path,
    figures_dir: Path,
    report_figures_dir: Path,
    tables_dir: Path,
    *,
    require_test: bool = False,
    require_bench: bool = False,
) -> None:
    evaluations = _load_evaluations(results_root)
    if require_test:
        expected = {f"test_{name}" for name in MODEL_ORDER}
        if not expected.issubset(evaluations):
            raise FileNotFoundError(
                f"Final test metrics are missing: {sorted(expected - evaluations.keys())}"
            )
        _write_error_table(
            results_root,
            tables_dir,
            required=True,
        )
    else:
        _write_error_table(results_root, tables_dir, required=False)
    _make_evaluation_figures(evaluations, figures_dir, report_figures_dir)
    _make_training_figure(results_root, figures_dir, report_figures_dir)
    _write_evaluation_tables(evaluations, tables_dir)
    _write_data_table_and_macros(results_root, evaluations, tables_dir)
    _write_training_config_table(tables_dir)

    bench_dir = results_root / "bench"
    kv_files = _load_kv_files(bench_dir)
    attention_files = list(bench_dir.rglob("attention_accuracy.csv"))
    speculative_files = list(bench_dir.rglob("speculative.json"))
    if require_bench:
        if not {"kv_1p5b", "kv_0p5b"}.issubset({name for name, _ in kv_files}):
            raise FileNotFoundError("Required 1.5B and 0.5B KV benchmark outputs are missing")
        if not attention_files:
            raise FileNotFoundError("Attention accuracy output is missing")
        speculative_runs = {path.parent.name for path in speculative_files}
        expected_speculative = {
            "speculative_base_0p5b",
            "speculative_finetuned_0p5b",
        }
        if not expected_speculative.issubset(speculative_runs):
            raise FileNotFoundError("Both speculative draft benchmark outputs are required")
    _make_kv_figures(kv_files, figures_dir, report_figures_dir, tables_dir)
    _make_attention_figures(bench_dir, figures_dir, report_figures_dir, tables_dir)
    _make_speculative_figures(bench_dir, figures_dir, report_figures_dir, tables_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "results" / "figures")
    parser.add_argument(
        "--report-figures-dir", type=Path, default=PROJECT_ROOT / "report" / "figures"
    )
    parser.add_argument("--tables-dir", type=Path, default=PROJECT_ROOT / "report" / "tables")
    parser.add_argument("--require-test", action="store_true")
    parser.add_argument("--require-bench", action="store_true")
    args = parser.parse_args()
    generate(
        args.results_root,
        args.figures_dir,
        args.report_figures_dir,
        args.tables_dir,
        require_test=args.require_test,
        require_bench=args.require_bench,
    )
    print(f"Generated figures in {args.figures_dir} and tables in {args.tables_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
