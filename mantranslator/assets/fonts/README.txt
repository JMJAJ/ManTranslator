Curated Font Library
====================

Drop open-licensed comic/manga fonts (.ttf / .otf) into this directory. The
renderer (`core/render.py`) scans this folder plus any fonts you add through
the GUI's font manager, then picks the closest match for each text region while
preserving the detected color, size and alignment.

Recommended open-licensed choices to add here:
  - "Comic Neue" (SIL OFL) - general speech
  - "Bangers" (SIL OFL) - shouting / emphasis, all-caps
  - "Patrick Hand" (SIL OFL) - handwritten style
  - A CJK-capable font (e.g. "Noto Sans JP/KR/SC", SIL OFL) for source-language
    fallbacks and mixed text

Fonts are intentionally not committed to keep the repository light and to avoid
bundling files whose licenses you have not reviewed. The app runs without any
fonts here by falling back to a built-in default, but results look best with a
small curated set.

---

What I added so far:
- Dialogue: Anime Ace or CC Wild Words.
- Narration Boxes: CC Astro City.
- Shouting & Emphasized Speech: Zud Juice.
- Sound Effects (SFX): CC Splashdown.