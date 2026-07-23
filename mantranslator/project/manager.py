"""Project persistence: create/open project folders and manage chapters.

A project lives in a single folder with this layout::

    <project>/
      project.json        # serialized Project (chapters, pages, regions)
      glossary.md         # cross-chapter names & translation memory
      <chapter>/
        source/           # imported original images
        output/           # rendered translated images

Images are copied into the project's ``source`` folder on import so the
project is self-contained and portable.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

from ..core.models import Chapter, Page, Project
from .glossary import Glossary


PROJECT_FILE = "project.json"
GLOSSARY_FILE = "glossary.md"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

# Images taller than this aspect ratio are treated as webtoon strips.
STRIP_ASPECT_RATIO = 3.0


class ProjectManager:
    """Owns the currently open project and mediates all disk operations."""

    def __init__(self) -> None:
        self.project: Project | None = None
        self.glossary: Glossary | None = None

    # ------------------------------------------------------------ lifecycle
    @property
    def root(self) -> Path | None:
        return Path(self.project.root) if self.project else None

    def create(self, parent_dir: str | Path, name: str,
               source_lang: str, target_lang: str) -> Project:
        root = Path(parent_dir) / _safe_name(name)
        root.mkdir(parents=True, exist_ok=True)
        self.project = Project(
            name=name,
            root=str(root),
            source_lang=source_lang,
            target_lang=target_lang,
        )
        self.glossary = Glossary.load(root / GLOSSARY_FILE)
        self.save()
        self.glossary.save()
        return self.project

    def open(self, root: str | Path) -> Project:
        root = Path(root)
        data = json.loads((root / PROJECT_FILE).read_text(encoding="utf-8"))
        self.project = Project.from_dict(data)
        # Keep the stored root in sync with the actual location.
        self.project.root = str(root)
        self.glossary = Glossary.load(root / GLOSSARY_FILE)
        return self.project

    def save(self) -> None:
        if not self.project:
            return
        root = Path(self.project.root)
        root.mkdir(parents=True, exist_ok=True)
        (root / PROJECT_FILE).write_text(
            json.dumps(self.project.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def save_glossary(self) -> None:
        if self.glossary:
            self.glossary.save()

    # ------------------------------------------------------------- chapters
    def add_chapter(self, name: str) -> Chapter:
        assert self.project is not None
        chapter = Chapter(name=_safe_name(name))
        (self._chapter_dir(chapter) / "source").mkdir(parents=True, exist_ok=True)
        (self._chapter_dir(chapter) / "output").mkdir(parents=True, exist_ok=True)
        self.project.chapters.append(chapter)
        self.save()
        return chapter

    def import_images(self, chapter: Chapter, image_paths: list[str | Path]) -> list[Page]:
        """Copy images into the chapter's source folder and register pages."""
        assert self.project is not None
        source_dir = self._chapter_dir(chapter) / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        added: list[Page] = []
        for src in sorted(image_paths, key=lambda p: str(p)):
            src = Path(src)
            if src.suffix.lower() not in IMAGE_EXTS:
                continue
            dest = _unique_dest(source_dir, src.name)
            shutil.copy2(src, dest)
            page = self._page_from_image(dest)
            chapter.pages.append(page)
            added.append(page)
        self.save()
        return added

    def output_path_for(self, chapter: Chapter, page: Page) -> Path:
        out_dir = self._chapter_dir(chapter) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{Path(page.image_path).stem}_translated.png"

    # -------------------------------------------------------------- helpers
    def _chapter_dir(self, chapter: Chapter) -> Path:
        assert self.project is not None
        return Path(self.project.root) / _safe_name(chapter.name)

    @staticmethod
    def _page_from_image(path: Path) -> Page:
        width = height = 0
        try:
            with Image.open(path) as img:
                width, height = img.size
        except OSError:
            pass
        is_strip = bool(height and width and height / max(width, 1) >= STRIP_ASPECT_RATIO)
        return Page(
            image_path=str(path),
            width=width,
            height=height,
            is_strip=is_strip,
        )


def _safe_name(name: str) -> str:
    """Return a filesystem-safe version of ``name``."""
    cleaned = "".join(c if c.isalnum() or c in " -_." else "_" for c in name).strip()
    return cleaned or "untitled"


def _unique_dest(directory: Path, filename: str) -> Path:
    dest = directory / filename
    if not dest.exists():
        return dest
    stem, suffix = Path(filename).stem, Path(filename).suffix
    i = 1
    while True:
        candidate = directory / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1
