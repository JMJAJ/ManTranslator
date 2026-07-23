"""Provider response-parsing tests (no network)."""
from mantranslator.ai.base import parse_translation_response


def test_parse_clean_json():
    raw = '{"translations": ["Hello", "World"], "new_terms": []}'
    result = parse_translation_response(raw, expected=2)
    assert result.translations == ["Hello", "World"]
    assert result.new_terms == []


def test_parse_json_in_code_fence():
    raw = '```json\n{"translations": ["A"], "new_terms": ' \
          '[{"source": "X", "translation": "Y", "type": "character"}]}\n```'
    result = parse_translation_response(raw, expected=1)
    assert result.translations == ["A"]
    assert result.new_terms[0]["source"] == "X"


def test_parse_pads_missing_items():
    raw = '{"translations": ["only one"]}'
    result = parse_translation_response(raw, expected=3)
    assert len(result.translations) == 3
    assert result.translations[0] == "only one"
    assert result.translations[1] == ""


def test_parse_truncates_extra_items():
    raw = '{"translations": ["a", "b", "c", "d"]}'
    result = parse_translation_response(raw, expected=2)
    assert result.translations == ["a", "b"]


def test_parse_falls_back_to_lines():
    raw = "First line\nSecond line"
    result = parse_translation_response(raw, expected=2)
    assert result.translations == ["First line", "Second line"]
