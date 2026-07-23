"""Data models for the translation pipeline.

These lightweight dataclasses are serializable to/from plain dicts so a project
and its per-page state can be persisted as JSON alongside the images.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Orientation(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class Alignment(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class PageStatus(str, Enum):
    PENDING = "pending"
    DETECTED = "detected"
    TRANSLATED = "translated"
    RENDERED = "rendered"
    ERROR = "error"


@dataclass
class TextRegion:
    """A single detected block of text on a page.

    Coordinates are in the page's full-resolution pixel space. ``polygon`` is a
    list of ``[x, y]`` points; ``bbox`` is the axis-aligned bounding box
    ``[x, y, w, h]`` derived from the polygon for convenience.
    """

    id: str
    bbox: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    polygon: list[list[int]] = field(default_factory=list)
    reading_order: int = 0
    orientation: str = Orientation.HORIZONTAL.value
    source_text: str = ""
    translated_text: str = ""
    ocr_confidence: float = 0.0
    used_vision_fallback: bool = False
    # Styling detected from the original / chosen for re-rendering.
    text_color: list[int] = field(default_factory=lambda: [0, 0, 0])
    stroke_color: list[int] = field(default_factory=lambda: [255, 255, 255])
    font_size: int = 24
    font_name: str = ""
    alignment: str = Alignment.CENTER.value
    manual_override: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextRegion":
        known = {k: data[k] for k in cls.__annotations__ if k in data}
        return cls(**known)


@dataclass
class Page:
    """A single comic page or webtoon strip."""

    image_path: str
    width: int = 0
    height: int = 0
    is_strip: bool = False
    status: str = PageStatus.PENDING.value
    regions: list[TextRegion] = field(default_factory=list)
    output_path: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["regions"] = [r.to_dict() for r in self.regions]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Page":
        regions = [TextRegion.from_dict(r) for r in data.get("regions", [])]
        known = {
            k: data[k]
            for k in cls.__annotations__
            if k in data and k != "regions"
        }
        return cls(regions=regions, **known)


@dataclass
class Chapter:
    """An ordered collection of pages."""

    name: str
    pages: list[Page] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "pages": [p.to_dict() for p in self.pages]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chapter":
        pages = [Page.from_dict(p) for p in data.get("pages", [])]
        return cls(name=data.get("name", "Chapter"), pages=pages)


@dataclass
class Project:
    """Top-level unit of work: a series with chapters and a shared glossary."""

    name: str
    root: str
    source_lang: str = "ja"
    target_lang: str = "en"
    chapters: list[Chapter] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root": self.root,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "chapters": [c.to_dict() for c in self.chapters],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Project":
        chapters = [Chapter.from_dict(c) for c in data.get("chapters", [])]
        return cls(
            name=data.get("name", "Untitled"),
            root=data.get("root", ""),
            source_lang=data.get("source_lang", "ja"),
            target_lang=data.get("target_lang", "en"),
            chapters=chapters,
        )
