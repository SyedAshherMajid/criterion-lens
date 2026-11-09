"""Tokenize expert examples with loss only on the assistant response."""

import logging
from collections.abc import Mapping
from typing import Any

from criterion_lens.data.prompts import build_messages, format_target
from criterion_lens.data.sentences import parse_numbered_note

LOGGER = logging.getLogger(__name__)
END_TOKEN = "<|im_end|>"


def _encode_messages_and_target(
    tokenizer: Any, messages: list[dict[str, str]], target: str
) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    end_token_id = tokenizer.convert_tokens_to_ids(END_TOKEN)
    if end_token_id is None or end_token_id == tokenizer.unk_token_id:
        raise ValueError(f"Tokenizer does not recognize the {END_TOKEN} token")
    return prompt_ids, [*target_ids, end_token_id]


def encode_example(
    tokenizer: Any,
    example: Mapping[str, Any],
    max_len: int,
    truncation_log: list[dict[str, Any]] | None = None,
) -> dict[str, list[int]]:
    """Encode one training example, masking the prompt and preserving target tokens.

    If needed, whole trailing note sentences are removed only while they are not
    gold evidence. A caller-provided list receives structured truncation records.
    """
    if max_len <= 0:
        raise ValueError("max_len must be positive")

    required = (
        "annotation_id",
        "note",
        "criterion_type",
        "criterion_text",
        "expert_sentences",
        "expert_eligibility",
    )
    missing = [key for key in required if key not in example]
    if missing:
        raise ValueError("Example is missing required fields: " + ", ".join(missing))

    note = example["note"]
    if not isinstance(note, str):
        raise ValueError("Example note must be a string")
    sentences = parse_numbered_note(note)
    evidence_ids = set(example["expert_sentences"])
    sentence_ids = {sentence_id for sentence_id, _ in sentences}
    missing_evidence = evidence_ids - sentence_ids
    if missing_evidence:
        raise ValueError(
            f"Annotation {example['annotation_id']} references missing evidence "
            f"sentence IDs: {sorted(missing_evidence)}"
        )
    target = format_target(example["expert_sentences"], example["expert_eligibility"])

    def encode_prompt(current_sentences: list[tuple[int, str]]) -> tuple[list[int], list[int]]:
        messages = build_messages(
            current_sentences, example["criterion_type"], example["criterion_text"]
        )
        return _encode_messages_and_target(tokenizer, messages, target)

    removed_sentence_ids: list[int] = []
    prompt_ids, response_ids = encode_prompt(sentences)
    input_ids = [*prompt_ids, *response_ids]
    while len(input_ids) > max_len:
        if not sentences:
            raise ValueError(
                f"Annotation {example['annotation_id']} needs {len(input_ids)} tokens "
                f"even with an empty note; target cannot be truncated to max_len {max_len}"
            )
        if sentences[-1][0] in evidence_ids:
            raise ValueError(
                f"Annotation {example['annotation_id']} needs {len(input_ids)} tokens "
                f"but max_len is {max_len}; truncation would remove gold evidence"
            )
        sentence_id, _ = sentences.pop()
        removed_sentence_ids.append(sentence_id)
        prompt_ids, response_ids = encode_prompt(sentences)
        input_ids = [*prompt_ids, *response_ids]

    if removed_sentence_ids:
        record = {
            "annotation_id": example["annotation_id"],
            "removed_sentence_ids": removed_sentence_ids,
            "removed_gold_evidence": False,
        }
        if truncation_log is not None:
            truncation_log.append(record)
        LOGGER.warning(
            "Truncated annotation %s by removing trailing note sentences %s",
            example["annotation_id"],
            removed_sentence_ids,
        )

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + response_ids,
    }
