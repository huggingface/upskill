"""Adapter that lets a ``CliAgentExecutor`` stand in for a fast-agent agent.

Implements the narrow surface upskill's ``generate`` / ``test`` flows actually
use (``send`` for free-form text, ``structured`` for JSON-schema-shaped output).
This is what makes ``upskill generate "..." --model cli.claude-code`` work
end-to-end without an Anthropic/OpenAI API key.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, ValidationError

from upskill.executors.contracts import ExecutionRequest, ExecutionResult

if TYPE_CHECKING:
    from pathlib import Path

    from upskill.executors.cli_agent import CliAgentExecutor

ModelT = TypeVar("ModelT", bound=BaseModel)


# Strip a single leading and trailing markdown code fence (```json ... ```).
_CODE_FENCE_OPEN = re.compile(r"^\s*```(?:json|JSON)?\s*\n?")
_CODE_FENCE_CLOSE = re.compile(r"\n?\s*```\s*$")


def _strip_code_fence(text: str) -> str:
    """Remove a single wrapping markdown code fence from JSON-ish output."""
    stripped = _CODE_FENCE_OPEN.sub("", text)
    stripped = _CODE_FENCE_CLOSE.sub("", stripped)
    return stripped.strip()


def _extract_json_payload(text: str) -> str | None:
    """Best-effort extraction of the first JSON object/array from free-form text."""
    cleaned = _strip_code_fence(text)
    if not cleaned:
        return None
    if cleaned[0] in "{[":
        return cleaned
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end != -1 and end > start:
            return cleaned[start : end + 1]
    return None


def _build_structured_prompt(message: str, schema: object) -> str:
    """Append a JSON-schema instruction so CLIs know to emit pure JSON."""
    schema_text = json.dumps(schema, indent=2)
    return (
        f"{message}\n\n"
        "## Output format\n\n"
        "Respond with ONLY a single JSON value matching this JSON schema. "
        "Do not wrap the response in markdown code fences. Do not include any "
        "explanatory prose before or after the JSON.\n\n"
        f"```\n{schema_text}\n```"
    )


@dataclass(slots=True)
class CliFastAgentAdapter:
    """Drive a ``CliAgentExecutor`` as if it were a fast-agent generator agent.

    Only the methods upskill's generation flows touch (``send`` and
    ``structured``) are implemented. Each call goes through the executor as a
    one-shot ``ExecutionRequest`` (no skill bundle, no workspace files), so
    artifacts land alongside the rest of the run for traceability.
    """

    executor: CliAgentExecutor
    model: str
    fastagent_config_path: Path
    cards_source_dir: Path
    artifact_root: Path
    agent_name: str = "skill_gen"
    _call_index: list[int] = field(default_factory=lambda: [0])

    async def send(
        self,
        message: str,
        request_params: object | None = None,
    ) -> str:
        """Run a free-form prompt; return the assistant's text reply."""
        del request_params
        result = await self._invoke(message)
        return result.output_text or ""

    async def structured(
        self,
        message: str,
        model: type[ModelT],
        request_params: object | None = None,
    ) -> tuple[ModelT | None, str]:
        """Ask the CLI for a JSON-schema-shaped reply and validate it."""
        del request_params
        prompt = _build_structured_prompt(message, model.model_json_schema())
        result = await self._invoke(prompt)
        raw = result.output_text or ""
        candidate = _extract_json_payload(raw)
        if candidate is None:
            return None, raw
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return None, raw
        try:
            return model.model_validate(payload), raw
        except ValidationError:
            return None, raw

    async def _invoke(self, prompt: str) -> ExecutionResult:
        """Execute one CLI call against ``self.executor`` and return its result."""
        self._call_index[0] += 1
        index = self._call_index[0]
        request = ExecutionRequest(
            prompt=prompt,
            model=self.model,
            agent=self.agent_name,
            fastagent_config_path=self.fastagent_config_path,
            artifact_dir=self.artifact_root / f"{self.agent_name}_call_{index:03d}",
            cards_source_dir=self.cards_source_dir,
            label=f"{self.agent_name}-{index}",
            skill=None,
            workspace_files={},
            metadata={"phase": self.agent_name, "call_index": index},
        )
        handle = await self.executor.execute(request)
        return await self.executor.collect(handle)
