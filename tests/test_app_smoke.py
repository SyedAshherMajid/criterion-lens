import json

import gradio as gr
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from app.screening import attention_lab, compare_decoding_modes, screen_patient
from app.ui import _examples, _load_example, _screen, build_app
from criterion_lens.modeling.load_model import LoadedModel


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = None

    def apply_chat_template(self, messages, **kwargs):
        assert any("Patient age is 52." in message["content"] for message in messages)
        return torch.tensor([[1, 2, 3]])

    def decode(self, tokens, *, skip_special_tokens):
        return json.dumps({"evidence": [0], "label": "included"})


def _tiny_loaded_model() -> LoadedModel:
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
        )
    ).eval()
    model.generate = lambda input_ids, **kwargs: torch.cat(
        [input_ids, input_ids.new_tensor([[4]])],
        dim=1,
    )
    return LoadedModel(
        model=model,
        tokenizer=TinyTokenizer(),
        model_id="tiny-local",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_screening_smoke_uses_tiny_model_and_exports_reports(tmp_path) -> None:
    loaded = _tiny_loaded_model()
    result = screen_patient(
        "0. Patient age is 52.",
        "Inclusion Criteria:\n- Adults aged 18 years or older.",
        model_id="tiny-local",
        decoding="no_cache",
        model_loader=lambda model_id: loaded,
        output_root=tmp_path,
    )

    assert result["verdict"] == "potentially eligible (preliminary)"
    assert result["rows"][0][3:5] == ["included", "0"]
    assert json.loads(result["report_json"])["criteria"][0]["label"] == "included"
    assert result["saved_files"]["html"].is_file()
    assert result["saved_files"]["json"].is_file()
    assert result["saved_files"]["md"].is_file()


@pytest.mark.parametrize(
    ("note", "criteria"),
    [
        ("", "Inclusion Criteria:\n- Adults."),
        ("0. Patient age is 52.", ""),
        ("0. Patient age is 52.", "Adults aged 18 or older."),
    ],
)
def test_screening_rejects_invalid_input_before_model_load(
    note, criteria, tmp_path
) -> None:
    def unexpected_model_load(*args, **kwargs):
        raise AssertionError("Invalid input must be rejected before model loading")

    with pytest.raises(ValueError):
        screen_patient(
            note,
            criteria,
            model_id="unused",
            model_loader=unexpected_model_load,
            output_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("note", "model_label", "message"),
    [
        ("", "Base Qwen2.5 0.5B", "Patient note must not be empty"),
        ("0. Patient age is 52.", "Unknown model", "Choose a supported model"),
    ],
)
def test_ui_returns_friendly_errors_for_invalid_input(note, model_label, message) -> None:
    with pytest.raises(gr.Error, match=message):
        _screen(
            note,
            "Inclusion Criteria:\n- Adults.",
            [],
            model_label,
            "greedy_kv",
            20,
            4,
        )


def test_decoding_comparison_reports_parity_and_speculative_acceptance(monkeypatch) -> None:
    calls = []

    def fake_screen_patient(note, criteria_text, *, decoding, **kwargs):
        calls.append(decoding)
        return {
            "rows": [["criterion-1", "inclusion", "Adults.", "included", "0"]],
            "decoder_stats": [{"proposed": 4, "accepted": 3}] if decoding == "speculative" else [],
        }

    monkeypatch.setattr("app.screening.screen_patient", fake_screen_patient)
    comparison = compare_decoding_modes(
        "0. Patient age is 52.",
        "Inclusion Criteria:\n- Adults.",
        model_id="tiny-local",
    )

    assert calls == ["no_cache", "greedy_kv", "speculative"]
    assert all(run["decision_and_evidence_equal_to_no_cache"] for run in comparison["runs"])
    assert comparison["runs"][2]["acceptance_rate"] == 0.75
    assert "not a benchmark" in comparison["note"]


def test_synthetic_examples_clear_editable_criteria() -> None:
    examples = _examples()
    note, criteria, rows = _load_example(examples[0]["name"])

    assert len(examples) == 3
    assert note == examples[0]["note"]
    assert criteria == examples[0]["criteria"]
    assert rows == []


def test_gradio_app_builds_all_tabs_and_attention_lab_runs() -> None:
    demo = build_app()

    assert isinstance(demo, gr.Blocks)
    result = attention_lab(
        sequence_length=8,
        head_dim=8,
        heads=2,
        block_size=4,
    )
    assert result["max_abs_error"] < 1e-5
    assert result["device"] == "cpu"
