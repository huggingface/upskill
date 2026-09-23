#!/usr/bin/env python3
"""Fake Kiro CLI for upskill end-to-end tests.

Mimics ``kiro-cli chat --no-interactive [...] <prompt>`` by emitting a
deterministic plain-text reply that includes the trailing positional prompt.
"""

from __future__ import annotations

import sys


def _extract_prompt(argv: list[str]) -> str:
    # The Kiro CLI invocation we build is:
    #   kiro-cli chat --no-interactive [provider.args...] <prompt>
    # So the trailing positional argument is always the prompt.
    if not argv:
        return ""
    return argv[-1]


def main() -> int:
    prompt = _extract_prompt(sys.argv[1:])
    sys.stdout.write(f"FAKE-KIRO: {prompt[:80]}\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
