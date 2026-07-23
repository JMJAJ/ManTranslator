# ManTranslator

AI-powered manga / webtoon translator with a dark-themed desktop GUI. It
detects text in comic images, reads it (OCR), translates it through a pluggable
local or cloud AI backend, erases the original text with AI inpainting, and
re-renders the translation in a matched font and color. A per-project markdown
glossary keeps character names and terminology consistent across chapters.

## Features

- Manga pages and tall webtoon strips (long strips are processed in overlapping tiles).
- Text detection via the vendored [comic-text-detector](https://github.com/dmMaze/comic-text-detector)
  (DBNet text mask + YOLO block detector), with PaddleOCR/OpenCV fallbacks.
- OCR: [manga-ocr](https://github.com/kha-white/manga-ocr) for Japanese,
  [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for Korean/Chinese/English,
  Tesseract as a fallback, plus an optional vision-LLM fallback for low-confidence regions.
- Translation backends: OpenAI-compatible (OpenAI/ChatGPT, LM Studio, Ollama,
  DeepSeek), Google Gemini, and Anthropic Claude.
- Text removal with LaMa AI inpainting (OpenCV Telea fallback).
- Curated font library with closest-match selection; preserves detected color,
  size and alignment; supports vertical Japanese layout.
- Interactive editor to review/correct each region (translation, font, size,
  color, alignment, orientation) and re-render locally without new AI calls.
- Cross-chapter consistency via an editable `glossary.md` (characters, terms,
  translation memory).

## Requirements

- Python 3.10+
- Windows, macOS or Linux
- Optional NVIDIA GPU (CUDA) for faster inpainting/detection; CPU works too.
- Tesseract installed on the system if you use the Tesseract OCR engine.

## Installation

```bash
# 1) Clone the upstream model repositories used by the pipeline
mkdir -p .repos
git -C .repos clone --depth 1 https://github.com/dmMaze/comic-text-detector.git
git -C .repos clone --depth 1 https://github.com/kha-white/manga-ocr.git
git -C .repos clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git

# 2) Create a virtual environment and install dependencies
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt

# 3) Install LaMa inpainting without its stale dependency pin.
#    simple-lama-inpainting's metadata pins pillow<10, which conflicts with
#    manga-ocr (pillow>=10); it runs fine on Pillow 10, so install it --no-deps.
pip install --no-deps simple-lama-inpainting==0.1.2
```

For GPU acceleration, install the CUDA build of PyTorch that matches your
system from https://pytorch.org before installing the rest of the requirements.

> Using `uv`? The same two steps apply:
> `uv pip install -r requirements.txt` then
> `uv pip install --no-deps simple-lama-inpainting==0.1.2`.

## Model weights

Weights download automatically on first use into the app's cache directory:

- `models/` under your OS cache dir (Windows: `%LOCALAPPDATA%\ManTranslator\models`).
- comic-text-detector weights (`comictextdetector.pt`) are fetched from the
  manga-image-translator release the first time detection runs.
- manga-ocr and PaddleOCR download their own models on first use.
- LaMa weights download on first inpaint.

The first translation run is therefore slower while models download and load.

## Running

```bash
python -m mantranslator
```

## First-time setup in the app

1. Open the **Settings** tab and click **Add...** under AI Providers.
   - Pick a preset (LM Studio, Ollama, OpenAI, DeepSeek, Gemini, Claude) or
     configure a custom endpoint.
   - Enter the model name and, for cloud services, your API key.
   - Click **Test Connection** to verify.
2. Choose the **Translation model** and, optionally, a vision-capable provider
   as the **Vision-OCR fallback**.
3. Set the default source/target languages, OCR engine and inpainting option.

### Provider notes

- **LM Studio**: start its local server; base URL `http://localhost:1234/v1`.
- **Ollama**: run `ollama serve`; base URL `http://localhost:11434/v1`; set the
  model (e.g. `llama3.1`). Use a vision model (e.g. `llava`) for vision OCR.
- **OpenAI / DeepSeek**: use their base URLs and an API key.
- **Gemini / Claude**: native SDKs; require an API key. These are vision-capable.

## Typical workflow

1. **New** project (choose a folder, name, source and target language).
2. **Add Chapter**, then **Import Images...** (pages or webtoon strips).
3. Select a page and click **Translate Selected Page**, or **Translate Chapter**.
4. Review results in the **Editor**: click a region to edit its translation,
   font, size, color, alignment or orientation, then **Apply & Re-render Page**.
5. **Export Page...** to save a translated image.
6. Edit names and terms anytime in the **Glossary** tab; they are reused on
   every subsequent translation for consistency.

Projects are self-contained folders:

```
<project>/
  project.json      # chapters, pages, detected regions and translations
  glossary.md       # characters, terms, translation memory
  <chapter>/
    source/         # imported originals
    output/         # rendered translations
```

## Development

```bash
pip install -r requirements.txt
python -m pytest        # runs the unit tests (no network/models required)
```

## Notes and licensing

- The vendored repositories in `.repos/` and their model weights carry their own
  licenses; review them before redistribution.
- Add only open-licensed fonts to `mantranslator/assets/fonts/`.
- API keys are stored locally in the app's config file
  (Windows: `%APPDATA%\ManTranslator\settings.json`).
