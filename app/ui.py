"""Gradio interface for the CriterionLens research prototype."""

import html
import json
from pathlib import Path
from typing import Any

import gradio as gr
from huggingface_hub.errors import HfHubHTTPError

from app.screening import (
    DISCLAIMER,
    MODEL_OPTIONS,
    attention_lab,
    compare_decoding_modes,
    screen_patient,
)
from criterion_lens.data.sentences import split_sentences
from criterion_lens.trials.criteria_parser import parse_criteria_text
from criterion_lens.trials.ctgov import ClinicalTrialsError, fetch_eligibility_criteria

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_PATH = Path(__file__).resolve().parent / "examples" / "demo_cases.json"


def _examples() -> list[dict[str, str]]:
    examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
    if not isinstance(examples, list) or not all(isinstance(item, dict) for item in examples):
        raise ValueError(f"Synthetic examples must be a JSON list: {EXAMPLES_PATH}")
    return examples


def _load_example(name: str) -> tuple[str, str, list[list[str]]]:
    example = next((item for item in _examples() if item["name"] == name), None)
    if example is None:
        raise ValueError(f"Unknown synthetic example: {name}")
    return example["note"], example["criteria"], []


def _read_upload(path: str | None) -> str:
    if not path:
        raise gr.Error("Choose a UTF-8 .txt file first")
    try:
        return Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise gr.Error("Uploaded note is not valid UTF-8") from error
    except OSError as error:
        raise gr.Error(f"Could not read uploaded note: {error}") from error


def _parse_preview(criteria_text: str) -> list[list[str]]:
    try:
        return [
            [criterion.criterion_type, criterion.criterion_text]
            for criterion in parse_criteria_text(criteria_text)
        ]
    except ValueError as error:
        raise gr.Error(str(error)) from error


def _fetch_trial(nct_id: str) -> str:
    try:
        return fetch_eligibility_criteria(nct_id)
    except (ValueError, ClinicalTrialsError) as error:
        raise gr.Error(str(error)) from error


def _screen(
    note: str,
    criteria_text: str,
    criterion_rows: list[list[str]],
    model_label: str,
    decoding: str,
    max_criteria: int,
    speculative_k: int,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, list[list[str]], str, str, list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        if model_label not in MODEL_OPTIONS:
            raise ValueError("Choose a supported model")
        model_id = MODEL_OPTIONS[model_label]
        result = screen_patient(
            note,
            criteria_text,
            model_id=model_id,
            decoding=decoding,
            max_criteria=max_criteria,
            speculative_k=speculative_k,
            criteria_rows=criterion_rows or None,
            progress=progress,
        )
    except (ValueError, RuntimeError, OSError, HfHubHTTPError) as error:
        raise gr.Error(str(error)) from error
    saved_files = result["saved_files"]
    files = [str(saved_files[key]) for key in ("html", "json", "md")]
    return (
        (
            f"## {result['verdict']}\n\n"
            f"**Missing information:** "
            f"{', '.join(result['aggregation']['missing_information_criteria']) or 'None'}\n\n"
            f"**Unsupported decisions:** "
            f"{', '.join(result['aggregation']['unsupported_criteria']) or 'None'}"
        ),
        result["rows"],
        result["report_html"],
        result["report_json"],
        files,
        result["decoder_stats"],
        result["decoder_stats"],
    )


def _highlight_note(note: str, rows: list[list[str]], event: gr.SelectData) -> str:
    index = event.index[0] if isinstance(event.index, tuple) else event.index
    if not isinstance(index, int) or index < 0 or index >= len(rows):
        raise gr.Error("Select a criterion row to highlight its evidence")
    evidence_ids = {
        int(value.strip())
        for value in rows[index][4].split(",")
        if value.strip()
    }
    rendered = []
    for sentence_id, sentence in split_sentences(note):
        content = html.escape(f"{sentence_id}. {sentence}")
        if sentence_id in evidence_ids:
            rendered.append(f'<mark data-sentence-id="{sentence_id}">{content}</mark>')
        else:
            rendered.append(f"<span>{content}</span>")
    return "<br>".join(rendered)


def _attention(
    sequence_length: int,
    head_dim: int,
    heads: int,
    block_size: int,
    causal: bool,
) -> dict[str, Any]:
    try:
        return attention_lab(
            sequence_length=sequence_length,
            head_dim=head_dim,
            heads=heads,
            block_size=block_size,
            causal=causal,
        )
    except (ValueError, RuntimeError) as error:
        raise gr.Error(str(error)) from error


def _compare(
    note: str,
    criteria_text: str,
    criterion_rows: list[list[str]],
    model_label: str,
    max_criteria: int,
    speculative_k: int,
) -> dict[str, Any]:
    try:
        return compare_decoding_modes(
            note,
            criteria_text,
            model_id=MODEL_OPTIONS[model_label],
            max_criteria=max_criteria,
            speculative_k=speculative_k,
            criteria_rows=criterion_rows or None,
        )
    except (ValueError, RuntimeError, OSError, HfHubHTTPError) as error:
        raise gr.Error(str(error)) from error


def _offline_files() -> list[list[str]]:
    directories = (PROJECT_ROOT / "results" / "bench", PROJECT_ROOT / "results" / "eval")
    files = sorted(
        path
        for directory in directories
        if directory.is_dir()
        for path in directory.glob("*.json")
    )
    return [
        [
            str(path.relative_to(PROJECT_ROOT)),
            "Available; review its metadata before interpreting measurements.",
        ]
        for path in files
    ]


def build_app() -> gr.Blocks:
    examples = _examples()
    example_names = [item["name"] for item in examples]
    with gr.Blocks(title="CriterionLens") as demo:
        gr.Markdown("# CriterionLens\n" + DISCLAIMER)
        last_decoder_stats = gr.State([])

        with gr.Tab("Screen a patient"):
            gr.Markdown(DISCLAIMER)
            with gr.Row():
                example_choice = gr.Dropdown(
                    choices=example_names,
                    value=example_names[0],
                    label="Synthetic example",
                )
                load_example_button = gr.Button("Load example")
            with gr.Row():
                note = gr.Textbox(
                    label="Patient note",
                    lines=12,
                    placeholder="Paste a numbered patient note or upload a UTF-8 .txt file.",
                )
                upload = gr.File(label="Upload UTF-8 note", file_types=[".txt"], type="filepath")
            upload.change(_read_upload, inputs=upload, outputs=note)
            with gr.Row():
                nct_id = gr.Textbox(label="ClinicalTrials.gov NCT ID", placeholder="NCT01234567")
                fetch_button = gr.Button("Fetch eligibility criteria")
            criteria = gr.Textbox(
                label="Inclusion and exclusion criteria",
                lines=10,
                placeholder="Inclusion Criteria:\n- ...\nExclusion Criteria:\n- ...",
            )
            fetch_button.click(_fetch_trial, inputs=nct_id, outputs=criteria)
            parse_button = gr.Button("Parse criteria into editable rows")
            criterion_rows = gr.Dataframe(
                headers=["type", "criterion text"],
                datatype=["str", "str"],
                row_count=(0, "dynamic"),
                col_count=(2, "fixed"),
                type="array",
                interactive=True,
                label="Editable criteria table",
            )
            parse_button.click(_parse_preview, inputs=criteria, outputs=criterion_rows)
            with gr.Row():
                model = gr.Dropdown(
                    choices=list(MODEL_OPTIONS),
                    value="Base Qwen2.5 0.5B",
                    label="Model",
                )
                decoding = gr.Dropdown(
                    choices=["greedy_kv", "no_cache", "speculative"],
                    value="greedy_kv",
                    label="Decoding mode",
                )
                max_criteria = gr.Slider(1, 50, value=20, step=1, label="Maximum criteria")
                speculative_k = gr.Slider(1, 8, value=4, step=1, label="Speculative k")
            run_button = gr.Button("Run screening", variant="primary")
            verdict = gr.Markdown(label="Preliminary outcome")
            results = gr.Dataframe(
                headers=[
                    "ID",
                    "Type",
                    "Criterion",
                    "Label",
                    "Evidence IDs",
                    "Evidence text",
                    "Explanation",
                ],
                type="array",
                interactive=False,
                label="Criterion results",
            )
            note_evidence = gr.HTML(label="Selected criterion evidence")
            results.select(_highlight_note, inputs=[note, results], outputs=note_evidence)
            report_html = gr.HTML(label="Printable report", padding=True)
            report_json = gr.Code(label="Machine-readable report", language="json")
            downloads = gr.File(label="Download report files", file_count="multiple")
            load_example_button.click(
                _load_example,
                inputs=example_choice,
                outputs=[note, criteria, criterion_rows],
            )

        with gr.Tab("Inference Lab"):
            gr.Markdown(DISCLAIMER)
            gr.Markdown(
                "Per-criterion diagnostics from the selected decoding mode appear in "
                "the first tab. "
                "Compare all three greedy modes on the same current note and criteria below."
            )
            decoder_stats = gr.JSON(label="Latest decoder diagnostics", value=[])
            compare_button = gr.Button("Compare decoding modes on current input")
            comparison = gr.JSON(label="Same-input decoding comparison")
            compare_button.click(
                _compare,
                inputs=[
                    note,
                    criteria,
                    criterion_rows,
                    model,
                    max_criteria,
                    speculative_k,
                ],
                outputs=comparison,
            )

        with gr.Tab("Attention Lab"):
            gr.Markdown(DISCLAIMER)
            gr.Markdown(
                "Compare the fp32 reference and tiled implementation on synthetic CPU inputs."
            )
            with gr.Row():
                sequence_length = gr.Slider(8, 256, value=64, step=8, label="Sequence length N")
                head_dim = gr.Slider(8, 128, value=32, step=8, label="Head dimension")
                heads = gr.Slider(1, 16, value=4, step=1, label="Heads")
                block_size = gr.Slider(4, 128, value=16, step=4, label="Tile size")
                causal = gr.Checkbox(value=True, label="Causal")
            attention_button = gr.Button("Compare attention")
            attention_output = gr.JSON(label="Synthetic comparison")
            attention_button.click(
                _attention,
                inputs=[sequence_length, head_dim, heads, block_size, causal],
                outputs=attention_output,
            )

        with gr.Tab("Offline results"):
            gr.Markdown(DISCLAIMER)
            gr.Markdown(
                "Only saved experiment files are listed; this panel does not create "
                "or infer results."
            )
            offline = gr.Dataframe(
                headers=["File", "Status"],
                datatype=["str", "str"],
                value=_offline_files(),
                interactive=False,
            )
            refresh = gr.Button("Refresh saved result list")
            refresh.click(_offline_files, outputs=offline)

        with gr.Tab("About"):
            gr.Markdown(DISCLAIMER)
            gr.Markdown(
                "CriterionLens is a research prototype for evidence-grounded clinical-trial "
                "criterion screening. Predictions are preliminary, sentence citations are "
                "model-generated, and final eligibility decisions require qualified human review. "
                "The examples shipped with this app are synthetic."
            )

        run_button.click(
            _screen,
            inputs=[
                note,
                criteria,
                criterion_rows,
                model,
                decoding,
                max_criteria,
                speculative_k,
            ],
            outputs=[
                verdict,
                results,
                report_html,
                report_json,
                downloads,
                last_decoder_stats,
                decoder_stats,
            ],
        )

    return demo
