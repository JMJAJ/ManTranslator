"""Prompt construction for translation and vision-OCR requests."""
from __future__ import annotations

# Human-readable language names used to make prompts clearer to the model.
LANG_NAMES = {
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "ru": "Russian",
    "it": "Italian",
    "id": "Indonesian",
    "vi": "Vietnamese",
}


def lang_name(code: str) -> str:
    return LANG_NAMES.get(code, code)


def translation_system_prompt(source_lang: str, target_lang: str,
                              glossary_block: str) -> str:
    """Build the system prompt for a batch translation request."""
    src = lang_name(source_lang)
    tgt = lang_name(target_lang)
    parts = [
        f"You are a professional comic/manga translator from {src} to {tgt}.",
        "Translate speech and narration naturally, matching tone, register and "
        "any shouting/whispering implied by the source. Keep translations "
        "concise enough to fit inside speech bubbles.",
        "Preserve honorifics only when they carry meaning; otherwise localize "
        "naturally. Do not add notes or explanations.",
        "You MUST honor the glossary: use the exact provided translation for "
        "every listed name or term.",
    ]
    if glossary_block:
        parts.append("\nGlossary:\n" + glossary_block)
    parts.append(
        "\nYou will receive a JSON array of source strings. Respond with ONLY a "
        "JSON object of this shape and nothing else:\n"
        '{\n'
        '  "translations": ["<translation for item 0>", "..."],\n'
        '  "new_terms": [\n'
        '    {"source": "<name/term>", "translation": "<chosen>", '
        '"type": "character|term", "notes": "<short>"}\n'
        '  ]\n'
        "}\n"
        "The 'translations' array MUST have exactly one entry per input item, in "
        "order. Put any newly encountered proper nouns worth keeping consistent "
        "into 'new_terms' (may be empty)."
    )
    return "\n".join(parts)


def translation_user_prompt(texts: list[str], context: str = "") -> str:
    import json

    payload = json.dumps(texts, ensure_ascii=False)
    prefix = f"Context: {context}\n\n" if context else ""
    return f"{prefix}Translate this JSON array of {len(texts)} strings:\n{payload}"


def vision_ocr_prompt(source_lang: str) -> str:
    """System/instruction prompt for reading text out of an image crop."""
    src = lang_name(source_lang)
    return (
        f"You are an OCR engine for {src} comic text. Read all {src} text in "
        "this image exactly as written, preserving line order top-to-bottom "
        "(right-to-left columns for vertical Japanese). Respond with ONLY the "
        "transcribed text, no quotes, labels or commentary. If there is no "
        "readable text, respond with an empty string."
    )
