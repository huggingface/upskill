#!/usr/bin/env python3
"""Fake GitHub Copilot CLI for upskill end-to-end tests.

Mimics ``copilot -p <prompt>`` by emitting a deterministic plain-text reply.
No usage/token fields, mirroring the real CLI's behavior in v1.
"""

from __future__ import annotations

import sys


def _extract_prompt(argv: list[str]) -> str:
    for index, value in enumerate(argv):
        if value == "-p" and index + 1 < len(argv):
            return argv[index + 1]
    return ""


def main() -> int:
    prompt = _extract_prompt(sys.argv[1:])
    sys.stdout.write(f"FAKE-COPILOT: {prompt[:80]}\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
