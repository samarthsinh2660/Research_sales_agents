"""Unit tests for extract_answer_text - handles Gemini's thinking-mode content shape."""
from agents.research.utils import extract_answer_text


def test_plain_string_passthrough():
    assert extract_answer_text("plain text") == "plain text"


def test_extracts_text_block_skips_thinking_block():
    content = [
        {"type": "thinking", "thinking": "internal reasoning, should be dropped"},
        {"type": "text", "text": "the actual answer"},
    ]
    assert extract_answer_text(content) == "the actual answer"


def test_joins_multiple_text_blocks():
    content = [
        {"type": "text", "text": "part one. "},
        {"type": "text", "text": "part two."},
    ]
    assert extract_answer_text(content) == "part one. part two."


def test_falls_back_when_no_typed_text_block():
    content = [{"foo": "bar", "text": "fallback text"}]
    assert extract_answer_text(content) == "fallback text"


def test_non_string_non_list_falls_back_to_str():
    assert extract_answer_text(42) == "42"
