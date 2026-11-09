import copy

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from criterion_lens.decoding.kv_cache import greedy_kv_cache
from criterion_lens.decoding.speculative import speculative_greedy


def _tiny_model(seed: int) -> Qwen2ForCausalLM:
    torch.manual_seed(seed)
    return Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=97,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    ).eval()


def test_identical_target_and_draft_accept_every_proposal() -> None:
    target = _tiny_model(17)
    draft = copy.deepcopy(target)
    input_ids = torch.tensor([[3, 10, 11, 12]])
    greedy = greedy_kv_cache(target, input_ids, max_new=10)

    result = speculative_greedy(target, draft, input_ids, max_new=10, k=3)

    assert torch.equal(result.tokens, greedy.tokens)
    assert result.proposed == result.accepted
    assert all(accepted == 3 for accepted in result.accepted_per_iteration[:-1])
    assert result.target_forwards == result.iterations + 1
    assert result.draft_forwards >= result.iterations + 1


def test_random_draft_still_matches_target_greedy_and_resynchronizes_caches() -> None:
    target = _tiny_model(27)
    draft = _tiny_model(81)
    input_ids = torch.tensor([[8, 19, 20, 21, 22]])
    greedy = greedy_kv_cache(target, input_ids, max_new=12)

    result = speculative_greedy(target, draft, input_ids, max_new=12, k=4)

    assert torch.equal(result.tokens, greedy.tokens)
    assert result.proposed > 0
    assert result.accepted <= result.proposed // 4
    assert len(result.cache_lengths) == result.iterations
    for row in result.cache_lengths:
        assert row["target_before"] == row["sequence_before"] - 1
        assert row["target_after_verify"] == (
            row["target_before"] + row["proposal_count"] + 1
        )
        assert row["target_after_crop"] == row["sequence_after"] - 1
        assert row["draft_before"] <= row["sequence_before"] - 1
        assert row["draft_after_crop"] <= row["sequence_after"] - 1


def test_eos_and_max_new_edges() -> None:
    target = _tiny_model(44)
    draft = copy.deepcopy(target)
    input_ids = torch.tensor([[4, 5, 6]])
    first_token = int(greedy_kv_cache(target, input_ids, max_new=1).tokens.item())

    eos_result = speculative_greedy(
        target,
        draft,
        input_ids,
        max_new=8,
        k=4,
        eos_id=first_token,
    )
    empty_result = speculative_greedy(
        target,
        draft,
        input_ids,
        max_new=0,
        k=4,
    )
    limited_result = speculative_greedy(
        target,
        draft,
        input_ids,
        max_new=3,
        k=8,
    )

    assert eos_result.tokens.tolist() == [[first_token]]
    assert eos_result.iterations == 0
    assert eos_result.target_forwards == 1
    assert empty_result.tokens.shape == (1, 0)
    assert empty_result.target_forwards == empty_result.draft_forwards == 0
    assert limited_result.tokens.shape == (1, 3)


def test_speculative_decoder_rejects_invalid_k_and_vocab_ids() -> None:
    target = _tiny_model(101)
    draft = copy.deepcopy(target)
    input_ids = torch.tensor([[3, 4]])
    with pytest.raises(ValueError, match="positive integer"):
        speculative_greedy(target, draft, input_ids, max_new=2, k=0)
    with pytest.raises(ValueError, match="shared vocabulary"):
        speculative_greedy(
            target,
            draft,
            input_ids,
            max_new=2,
            k=1,
            eos_id=target.config.vocab_size,
        )
