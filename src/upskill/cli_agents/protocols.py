"""Narrow generator protocol used by upskill's skill/test-generation flows.

Both fast-agent's ``AgentProtocol`` and ``CliFastAgentAdapter`` structurally
satisfy this. Using a narrower protocol keeps ``generate.py`` decoupled from
the concrete agent implementation (fast-agent or CLI-backed).

The signatures here intentionally describe *only* the shape upskill calls, not
fast-agent's full API. Both implementations may accept additional optional
parameters; structural compatibility just requires they accept the calls
upskill actually makes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from pydantic import BaseModel

    ModelT = TypeVar("ModelT", bound=BaseModel)


@runtime_checkable
class SkillGenerator(Protocol):
    """Surface upskill calls on its skill_gen / test_gen agents."""

    async def send(self, message: str, /) -> str: ...

    async def structured(
        self,
        message: str,
        model: type[ModelT],
        /,
    ) -> tuple[ModelT | None, object]: ...
