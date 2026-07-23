"""ManTranslator entry point.

Run with ``python -m mantranslator`` or the installed ``mantranslator`` command.
"""
from __future__ import annotations

import sys


def main() -> int:
    from .gui.app import run

    return run()


if __name__ == "__main__":
    sys.exit(main())
