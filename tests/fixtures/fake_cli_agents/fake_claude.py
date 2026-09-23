#!/usr/bin/env python3
"""Fake Claude Code CLI for upskill end-to-end tests.

Mimics ``claude -p <prompt> --output-format json`` by emitting a JSON envelope
with a deterministic ``result`` and synthetic ``usage`` token counts.
"""

from __future__ import annotations

import json
import sys


def _extract_prompt(argv: list[str]) -> str:
    for index, value in enumerate(argv):
        if value == "-p" and index + 1 < len(argv):
            return argv[index + 1]
    return ""


def main() -> int:
    prompt = _extract_prompt(sys.argv[1:])
    payload = {
        "result": f"FAKE-CLAUDE: {prompt[:80]}",
        "usage": {
            "input_tokens": 13,
            "output_tokens": 7,
        },
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
