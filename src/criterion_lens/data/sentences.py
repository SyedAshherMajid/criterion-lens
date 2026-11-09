"""Split numbered clinical notes without changing their evidence sentence IDs."""

import logging
import re

LOGGER = logging.getLogger(__name__)
NUMBERED_SENTENCE = re.compile(r"(?<!\S)(\d+)\.\s")
FALLBACK_BOUNDARY = re.compile(
    r"(?<!Dr\.)(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Prof\.)(?<!St\.)(?<=[.!?])\s+",
    flags=re.IGNORECASE,
)


def parse_numbered_note(note: str) -> list[tuple[int, str]]:
    """Parse dataset-style notes and require zero-based contiguous sentence IDs."""
    candidates = list(NUMBERED_SENTENCE.finditer(note))
    if not candidates:
        raise ValueError("Note does not contain numbered sentences")
    if note[: candidates[0].start(1)].strip():
        raise ValueError("Unexpected text before the first numbered sentence")

    matches = []
    for index, candidate in enumerate(candidates):
        body_end = (
            candidates[index + 1].start(1)
            if index + 1 < len(candidates)
            else len(note)
        )
        sentence_id = int(candidate.group(1))
        body = note[candidate.end() : body_end].strip()
        expected_id = len(matches)
        if sentence_id == expected_id:
            if not body:
                raise ValueError(f"Sentence {sentence_id} has no text")
            matches.append(candidate)
        elif sentence_id > expected_id and not body:
            continue
        else:
            received_ids = [int(match.group(1)) for match in matches] + [sentence_id]
            raise ValueError(
                f"Sentence IDs must be contiguous from zero; received {received_ids}"
            )

    sentences: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start(1) if index + 1 < len(matches) else len(note)
        sentence_id = int(match.group(1))
        sentence = note[match.end() : end].strip()
        if not sentence:
            raise ValueError(f"Sentence {sentence_id} has no text")
        sentences.append((sentence_id, sentence))
    return sentences


def _fallback_split(text: str) -> list[str]:
    """Split at simple punctuation boundaries while protecting common abbreviations."""
    return [segment.strip() for segment in FALLBACK_BOUNDARY.split(text) if segment.strip()]


def split_sentences(text: str) -> list[tuple[int, str]]:
    """Split a note into zero-based sentence IDs, preserving supplied IDs when present.

    Numbered dataset notes are parsed directly so evidence IDs remain stable. Unnumbered
    app notes use pysbd for abbreviation-aware boundaries, with a conservative local
    regex fallback when that optional package is unavailable.
    """
    stripped_text = text.strip()
    if not stripped_text:
        return []
    if NUMBERED_SENTENCE.search(stripped_text):
        return parse_numbered_note(stripped_text)

    try:
        import pysbd
    except ImportError:
        LOGGER.warning("pysbd is unavailable; using the basic sentence-boundary fallback")
        segments = _fallback_split(stripped_text)
    else:
        segmenter = pysbd.Segmenter(language="en", clean=False)
        segments = [segment.strip() for segment in segmenter.segment(stripped_text)]
        if not segments:
            LOGGER.warning("pysbd returned no sentences; using the basic fallback")
            segments = _fallback_split(stripped_text)

    return [(sentence_id, sentence) for sentence_id, sentence in enumerate(segments)]
