import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache

from criterion_lens.decoding.kv_cache import greedy_kv_cache, greedy_no_cache


@pytest.fixture(scope="module")
def tiny_qwen() -> Qwen2ForCausalLM:
    torch.manual_seed(2026)
    config = Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        eos_token_id=2,
    )
    return Qwen2ForCausalLM(config).eval()


@pytest.mark.parametrize("prompt_length", [2, 4, 7])
@pytest.mark.parametrize("max_new", [1, 2, 4])
def test_kv_and_no_cache_generate_identical_tokens(
    tiny_qwen: Qwen2ForCausalLM,
    prompt_length: int,
    max_new: int,
) -> None:
    input_ids = torch.arange(3, 3 + prompt_length).unsqueeze(0)

    reference = greedy_no_cache(tiny_qwen, input_ids, max_new)
    cached = greedy_kv_cache(tiny_qwen, input_ids, max_new)

    assert torch.equal(cached.tokens, reference.tokens)
    assert cached.tokens.shape == (1, max_new)
    assert cached.prefill_s >= 0
    assert cached.decode_s >= 0
    assert len(cached.per_step) == max_new
    assert [step["cache_len"] for step in cached.per_step] == list(
        range(prompt_length, prompt_length + max_new)
    )
    assert [step["processed_tokens"] for step in cached.per_step] == [
        step["cache_len"] for step in cached.per_step
    ]
    expected_bytes_per_token = (
        2
        * tiny_qwen.config.num_hidden_layers
        * tiny_qwen.config.num_key_value_heads
        * tiny_qwen.config.hidden_size
        // tiny_qwen.config.num_attention_heads
        * torch.tensor([], dtype=next(tiny_qwen.parameters()).dtype).element_size()
    )
    assert [step["cache_bytes"] for step in cached.per_step] == [
        expected_bytes_per_token * length
        for length in range(prompt_length, prompt_length + max_new)
    ]


def test_no_cache_records_zero_cache_and_empty_generation(
    tiny_qwen: Qwen2ForCausalLM,
) -> None:
    input_ids = torch.tensor([[3, 4, 5]])

    result = greedy_no_cache(tiny_qwen, input_ids, 2)
    empty = greedy_kv_cache(tiny_qwen, input_ids, 0)

    assert all(step["cache_len"] == step["cache_bytes"] == 0 for step in result.per_step)
    assert empty.tokens.shape == (1, 0)
    assert empty.prefill_s == empty.decode_s == 0
    assert empty.per_step == []


def test_wrong_decode_position_does_not_match_no_cache_reference(
    tiny_qwen: Qwen2ForCausalLM,
) -> None:
    input_ids = torch.tensor([[90, 81]])
    reference = greedy_no_cache(tiny_qwen, input_ids, 16)
    cache = DynamicCache(config=tiny_qwen.config)
    attention_mask = torch.ones_like(input_ids)
    prompt_positions = torch.arange(input_ids.shape[1]).unsqueeze(0)

    with torch.inference_mode():
        output = tiny_qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=prompt_positions,
            cache_position=torch.arange(input_ids.shape[1]),
            past_key_values=cache,
            use_cache=True,
        )
        generated = [output.logits[:, -1, :].argmax(dim=-1, keepdim=True)]
        for _ in range(15):
            current_length = cache.get_seq_length()
            output = tiny_qwen(
                input_ids=generated[-1],
                attention_mask=torch.ones((1, current_length + 1), dtype=torch.long),
                position_ids=torch.zeros((1, 1), dtype=torch.long),
                cache_position=torch.tensor([current_length]),
                past_key_values=cache,
                use_cache=True,
            )
            generated.append(output.logits[:, -1, :].argmax(dim=-1, keepdim=True))

    wrong_positions = torch.cat(generated, dim=1)
    assert not torch.equal(wrong_positions, reference.tokens)


@pytest.mark.parametrize(
    ("input_ids", "max_new", "eos_id"),
    [
        (torch.empty((0, 3), dtype=torch.long), 1, None),
        (torch.empty((2, 3), dtype=torch.long), 1, None),
        (torch.tensor([[1, 2]]), -1, None),
        (torch.tensor([[1, 2]]), 1, -1),
    ],
)
def test_generation_rejects_invalid_inputs(
    tiny_qwen: Qwen2ForCausalLM,
    input_ids: torch.Tensor,
    max_new: int,
    eos_id: int | None,
) -> None:
    with pytest.raises(ValueError):
        greedy_kv_cache(tiny_qwen, input_ids, max_new, eos_id)
