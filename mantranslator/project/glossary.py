"""Per-project glossary stored as a human-editable markdown file.

The glossary keeps translations consistent across chapters. It has three
sections backed by markdown tables:

* Characters - proper names with a chosen translation and notes.
* Terms & Places - other recurring nouns that must stay consistent.
* Translation Memory - exact source lines mapped to their translation, used to
  short-circuit repeated phrases and keep them identical.

The file is parsed leniently: unknown sections and free-form prose between
tables are preserved on load only insofar as the structured entries are
concerned. On save the file is fully regenerated from the structured data, so
edits should be made to table rows rather than surrounding prose.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import re


CHARACTERS_HEADING = "## Characters"
TERMS_HEADING = "## Terms & Places"
MEMORY_HEADING = "## Translation Memory"


@dataclass
class GlossaryEntry:
    source: str
    translation: str
    romanization: str = ""
    notes: str = ""


@dataclass
class Glossary:
    """In-memory representation of a project's glossary markdown file."""

    path: Path
    characters: list[GlossaryEntry] = field(default_factory=list)
    terms: list[GlossaryEntry] = field(default_factory=list)
    memory: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: str | Path) -> "Glossary":
        path = Path(path)
        gloss = cls(path=path)
        if not path.exists():
            return gloss

        section: str | None = None
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.rstrip()
            stripped = line.strip()
            if stripped.startswith("## "):
                if stripped == CHARACTERS_HEADING:
                    section = "characters"
                elif stripped == TERMS_HEADING:
                    section = "terms"
                elif stripped == MEMORY_HEADING:
                    section = "memory"
                else:
                    section = None
                continue
            if not stripped.startswith("|"):
                continue
            cells = _parse_row(stripped)
            if cells is None:  # header or separator row
                continue
            if section == "characters":
                gloss.characters.append(_entry_from_cells(cells, with_rom=True))
            elif section == "terms":
                gloss.terms.append(_entry_from_cells(cells, with_rom=True))
            elif section == "memory" and len(cells) >= 2:
                src, dst = cells[0], cells[1]
                if src:
                    gloss.memory[src] = dst
        return gloss

    # ------------------------------------------------------------------ save
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.render(), encoding="utf-8")

    def render(self) -> str:
        lines: list[str] = ["# Glossary", ""]
        lines.append(
            "Edit the table rows below to control naming and phrasing. Rows are "
            "reused automatically on every translation to keep chapters "
            "consistent."
        )
        lines.append("")

        lines.append(CHARACTERS_HEADING)
        lines.append("")
        lines.append("| Source | Romanization | Translation | Notes |")
        lines.append("| --- | --- | --- | --- |")
        for e in self.characters:
            lines.append(
                f"| {_esc(e.source)} | {_esc(e.romanization)} | "
                f"{_esc(e.translation)} | {_esc(e.notes)} |"
            )
        lines.append("")

        lines.append(TERMS_HEADING)
        lines.append("")
        lines.append("| Source | Romanization | Translation | Notes |")
        lines.append("| --- | --- | --- | --- |")
        for e in self.terms:
            lines.append(
                f"| {_esc(e.source)} | {_esc(e.romanization)} | "
                f"{_esc(e.translation)} | {_esc(e.notes)} |"
            )
        lines.append("")

        lines.append(MEMORY_HEADING)
        lines.append("")
        lines.append("| Source | Translation |")
        lines.append("| --- | --- |")
        for src, dst in self.memory.items():
            lines.append(f"| {_esc(src)} | {_esc(dst)} |")
        lines.append("")

        return "\n".join(lines) + "\n"

    # --------------------------------------------------------------- mutate
    def add_character(self, entry: GlossaryEntry) -> bool:
        return _merge_entry(self.characters, entry)

    def add_term(self, entry: GlossaryEntry) -> bool:
        return _merge_entry(self.terms, entry)

    def remember(self, source: str, translation: str) -> None:
        source = source.strip()
        if source:
            self.memory[source] = translation.strip()

    # --------------------------------------------------------------- prompt
    def as_prompt_block(self, max_entries: int = 200) -> str:
        """Render a compact glossary block for injection into an LLM prompt."""
        parts: list[str] = []
        named = self.characters + self.terms
        if named:
            parts.append("Names and terms (source => required translation):")
            for e in named[:max_entries]:
                extra = f" [{e.notes}]" if e.notes else ""
                parts.append(f"- {e.source} => {e.translation}{extra}")
        if self.memory:
            parts.append("")
            parts.append("Previously translated lines (reuse verbatim if identical):")
            for src, dst in list(self.memory.items())[:max_entries]:
                parts.append(f"- {src} => {dst}")
        return "\n".join(parts).strip()


# --------------------------------------------------------------------- helpers
def _parse_row(line: str) -> list[str] | None:
    """Parse a markdown table row into cells, or ``None`` for header/separator."""
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|") and not inner.endswith("\\|"):
        inner = inner[:-1]
    # Split on unescaped pipes only, so ``\|`` stays inside a cell.
    cells = [c.strip() for c in re.split(r"(?<!\\)\|", inner)]
    # Separator rows look like ``--- | ---``.
    if all(set(c) <= {"-", ":", " "} and c for c in cells):
        return None
    # Skip the header row that labels columns.
    lowered = [c.lower() for c in cells]
    if lowered[:1] == ["source"]:
        return None
    return cells


def _entry_from_cells(cells: list[str], with_rom: bool) -> GlossaryEntry:
    cells = cells + [""] * 4  # pad
    if with_rom:
        return GlossaryEntry(
            source=_unesc(cells[0]),
            romanization=_unesc(cells[1]),
            translation=_unesc(cells[2]),
            notes=_unesc(cells[3]),
        )
    return GlossaryEntry(source=_unesc(cells[0]), translation=_unesc(cells[1]))


def _merge_entry(bucket: list[GlossaryEntry], entry: GlossaryEntry) -> bool:
    """Add ``entry`` if its source is new; returns True when added."""
    src = entry.source.strip()
    if not src:
        return False
    for existing in bucket:
        if existing.source.strip().lower() == src.lower():
            return False
    bucket.append(entry)
    return True


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").strip()


def _unesc(text: str) -> str:
    return text.replace("\\|", "|").replace("\\\\", "\\").strip()
