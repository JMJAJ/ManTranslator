"""Glossary markdown round-trip and prompt-block tests."""
from mantranslator.project.glossary import Glossary, GlossaryEntry


def test_glossary_round_trip(tmp_path):
    path = tmp_path / "glossary.md"
    gloss = Glossary.load(path)
    gloss.add_character(GlossaryEntry(source="田中", romanization="Tanaka",
                                      translation="Tanaka", notes="protagonist"))
    gloss.add_term(GlossaryEntry(source="魔法学院", translation="Magic Academy"))
    gloss.remember("こんにちは", "Hello")
    gloss.save()

    reloaded = Glossary.load(path)
    assert len(reloaded.characters) == 1
    assert reloaded.characters[0].source == "田中"
    assert reloaded.characters[0].translation == "Tanaka"
    assert reloaded.characters[0].notes == "protagonist"
    assert len(reloaded.terms) == 1
    assert reloaded.terms[0].translation == "Magic Academy"
    assert reloaded.memory["こんにちは"] == "Hello"


def test_glossary_dedupes_characters(tmp_path):
    gloss = Glossary(path=tmp_path / "g.md")
    assert gloss.add_character(GlossaryEntry(source="A", translation="Alpha"))
    # Same source (case-insensitive) is not added twice.
    assert not gloss.add_character(GlossaryEntry(source="a", translation="Other"))
    assert len(gloss.characters) == 1


def test_pipe_characters_are_escaped(tmp_path):
    path = tmp_path / "g.md"
    gloss = Glossary.load(path)
    gloss.add_term(GlossaryEntry(source="A|B", translation="C|D"))
    gloss.save()
    reloaded = Glossary.load(path)
    assert reloaded.terms[0].source == "A|B"
    assert reloaded.terms[0].translation == "C|D"


def test_prompt_block_contains_entries(tmp_path):
    gloss = Glossary(path=tmp_path / "g.md")
    gloss.add_character(GlossaryEntry(source="X", translation="Xavier"))
    gloss.remember("hi", "hello")
    block = gloss.as_prompt_block()
    assert "X => Xavier" in block
    assert "hi => hello" in block
