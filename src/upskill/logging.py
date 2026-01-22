"""Run logging and session tracking for upskill (similar to skills-test)."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from fast_agent import ConversationSummary
from fast_agent.constants import FAST_AGENT_TIMING, FAST_AGENT_USAGE
from fast_agent.mcp.helpers.content_helpers import get_text

from upskill.models import BatchSummary, ConversationStats, RunMetadata, RunResult, TestResult

# CSV field names for run summaries (matching skills-test format)
FIELDNAMES = [
    "batch_id",
    "run_number",
    "model",
    "task",
    "timestamp",
    "passed",
    "assertions_passed",
    "assertions_total",
    # Token metrics
    "input_tokens",
    "output_tokens",
    "total_tokens",
    # Timing metrics
    "conversation_span_ms",
    "llm_time_ms",
    "tool_time_ms",
    # Conversation metrics
    "turns",
    "tool_calls",
    "tool_errors",
    # Tool categorization
    "mcp_calls",
    "execute_calls",
    # Error and session info
    "error_message",
    "session_id",
    "session_history_file",
]


def create_batch_folder(runs_dir: Path) -> tuple[str, Path]:
    """Create a timestamped batch folder for run artifacts.

    Returns:
        Tuple of (batch_id, batch_folder_path)
    """
    batch_id = datetime.now().strftime("%Y_%m_%d_%H_%M")
    batch_folder = runs_dir / batch_id
    batch_folder.mkdir(parents=True, exist_ok=True)
    return batch_id, batch_folder


def create_run_folder(batch_folder: Path, run_number: int) -> Path:
    """Create a folder for a single run within a batch."""
    run_folder = batch_folder / f"run_{run_number}"
    run_folder.mkdir(exist_ok=True)
    return run_folder


def write_run_metadata(run_folder: Path, metadata: RunMetadata) -> None:
    """Write run metadata to JSON file."""
    payload = {
        "model": metadata.model,
        "task": metadata.task,
        "batch_id": metadata.batch_id,
        "run_number": metadata.run_number,
        "timestamp": metadata.timestamp.isoformat(),
    }
    path = run_folder / "run_metadata.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_run_metadata(run_folder: Path) -> dict:
    """Load run metadata from JSON file."""
    metadata_path = run_folder / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_run_result(run_folder: Path, result: RunResult) -> None:
    """Write complete run result to JSON file."""
    payload = result.model_dump(mode="json")
    path = run_folder / "run_result.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_run_result(run_folder: Path) -> RunResult | None:
    """Load run result from JSON file."""
    result_path = run_folder / "run_result.json"
    if not result_path.exists():
        return None
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
        return RunResult(**data)
    except (json.JSONDecodeError, ValueError):
        return None


def extract_tokens_from_messages(
    messages: list,
) -> tuple[int, int, int, list[dict[str, object]]]:
    """Extract token counts from message channels.

    FastAgent stores usage data in the fast-agent usage channel as JSON.
    """
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    usage_summaries: list[dict[str, object]] = []

    for msg in messages:
        channels = getattr(msg, "channels", None)
        if not channels:
            continue

        # Look for usage channel (fast-agent-usage)
        usage_content = channels.get(FAST_AGENT_USAGE, [])
        for content in usage_content:
            # Content may be TextContent or a serialized dict
            if isinstance(content, dict):
                text = content.get("text")
            else:
                text = get_text(content) or getattr(content, "text", None)
            if not text:
                continue
            try:
                usage_data = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                continue

            if not isinstance(usage_data, dict):
                continue

            turn_data = usage_data.get("turn")
            summary_data = usage_data.get("summary")

            if isinstance(turn_data, dict):
                input_tokens += int(turn_data.get("input_tokens", 0) or 0)
                output_tokens += int(turn_data.get("output_tokens", 0) or 0)
                total_tokens += int(turn_data.get("total_tokens", 0) or 0)
                if isinstance(summary_data, dict):
                    usage_summaries.append(summary_data)
                continue

            if isinstance(summary_data, dict):
                usage_summaries.append(summary_data)
                input_tokens = max(
                    input_tokens,
                    int(summary_data.get("cumulative_input_tokens", 0) or 0),
                )
                output_tokens = max(
                    output_tokens,
                    int(summary_data.get("cumulative_output_tokens", 0) or 0),
                )
                total_tokens = max(
                    total_tokens,
                    int(summary_data.get("cumulative_billing_tokens", 0) or 0),
                )

    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    return input_tokens, output_tokens, total_tokens, usage_summaries





def extract_timing_from_messages(messages: list) -> list[dict[str, object]]:
    """Extract timing payloads from message channels."""
    timings: list[dict[str, object]] = []
    for msg in messages:
        channels = getattr(msg, "channels", None)
        if not channels:
            continue
        timing_content = channels.get(FAST_AGENT_TIMING, [])
        for content in timing_content:
            if isinstance(content, dict):
                text = content.get("text")
            else:
                text = get_text(content) or getattr(content, "text", None)
            if not text:
                continue
            try:
                payload = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(payload, dict):
                timings.append(payload)
    return timings


def extract_stats_from_summary(summary: ConversationSummary) -> ConversationStats:
    """Extract conversation statistics from a FastAgent ConversationSummary.

    This uses FastAgent's built-in conversation analysis to get rich metrics
    including timing, tool categorization, and per-tool breakdowns.
    """
    # Get timing metrics (ConversationSummary returns milliseconds directly)
    conversation_span_ms = summary.conversation_span_ms or 0.0
    llm_time_ms = summary.total_elapsed_time_ms or 0.0
    # Tool time not directly available, estimate from difference
    tool_time_ms = max(0.0, conversation_span_ms - llm_time_ms)

    # Get tool call metrics
    tool_calls = summary.tool_calls or 0
    tool_errors = summary.tool_errors or 0

    # Categorize tool calls (MCP vs Execute)
    # MCP tools have "__" in their name (e.g., "filesystem__read")
    tool_call_map = dict(summary.tool_call_map or {})
    tool_error_map = dict(summary.tool_error_map or {})

    mcp_calls = sum(count for name, count in tool_call_map.items() if "__" in name)
    execute_calls = sum(count for name, count in tool_call_map.items() if "__" not in name)

    # Extract token metrics from message channels
    input_tokens, output_tokens, total_tokens, usage_summaries = extract_tokens_from_messages(
        summary.messages
    )
    timing_summaries = extract_timing_from_messages(summary.messages)

    return ConversationStats(
        conversation_span_ms=conversation_span_ms,
        llm_time_ms=llm_time_ms,
        tool_time_ms=tool_time_ms,
        turns=summary.turn_count or 0,
        tool_calls=tool_calls,
        tool_errors=tool_errors,
        mcp_calls=mcp_calls,
        execute_calls=execute_calls,
        tool_call_map=tool_call_map,
        tool_error_map=tool_error_map,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        usage_summaries=usage_summaries,
        timing_summaries=timing_summaries,
    )




def aggregate_conversation_stats(results: list[TestResult]) -> ConversationStats:
    """Aggregate ConversationStats across multiple test results."""
    aggregate = ConversationStats()
    for result in results:
        stats = result.stats
        aggregate.conversation_span_ms += stats.conversation_span_ms
        aggregate.llm_time_ms += stats.llm_time_ms
        aggregate.tool_time_ms += stats.tool_time_ms
        aggregate.turns += stats.turns
        aggregate.tool_calls += stats.tool_calls
        aggregate.tool_errors += stats.tool_errors
        aggregate.mcp_calls += stats.mcp_calls
        aggregate.execute_calls += stats.execute_calls
        aggregate.input_tokens += stats.input_tokens
        aggregate.output_tokens += stats.output_tokens
        aggregate.total_tokens += stats.total_tokens
        aggregate.usage_summaries.extend(stats.usage_summaries)
        aggregate.timing_summaries.extend(stats.timing_summaries)
        for name, count in stats.tool_call_map.items():
            aggregate.tool_call_map[name] = aggregate.tool_call_map.get(name, 0) + count
        for name, count in stats.tool_error_map.items():
            aggregate.tool_error_map[name] = aggregate.tool_error_map.get(name, 0) + count
    if aggregate.total_tokens == 0:
        aggregate.total_tokens = aggregate.input_tokens + aggregate.output_tokens
    return aggregate


def summarize_conversation_stats(history_path: Path | None) -> ConversationStats:
    """Extract conversation statistics from session history file.

    This is a fallback for when ConversationSummary is not available.
    Prefer using extract_stats_from_summary() when you have access to the agent.
    """
    if not history_path or not history_path.exists():
        return ConversationStats()

    try:
        data = json.loads(history_path.read_text(encoding="utf-8"))
        messages = data if isinstance(data, list) else data.get("messages", [])

        # Count turns (user messages)
        turns = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "user")

        # Try to extract token usage if available
        input_tokens = 0
        output_tokens = 0
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                usage = msg.get("usage", {})
                if usage:
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                    break

        return ConversationStats(
            turns=turns,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
    except (json.JSONDecodeError, ValueError):
        return ConversationStats()


def iter_run_folders(runs_folder: Path) -> list[Path]:
    """Iterate over all run folders in a runs directory."""
    run_folders: list[Path] = []
    if not runs_folder.exists():
        return run_folders

    for batch_dir in sorted(p for p in runs_folder.iterdir() if p.is_dir()):
        if not any(
            child.name.startswith("run_") for child in batch_dir.iterdir() if child.is_dir()
        ):
            continue
        for run_dir in sorted(batch_dir.glob("run_*")):
            if run_dir.is_dir():
                run_folders.append(run_dir)
    return run_folders


def summarize_runs_to_csv(runs_folder: Path, output_path: Path | None = None) -> Path:
    """Summarize all runs in a folder to a CSV file.

    Args:
        runs_folder: Path to the runs directory
        output_path: Path for output CSV (defaults to runs_folder/results.csv)

    Returns:
        Path to the generated CSV file
    """
    if output_path is None:
        output_path = runs_folder / "results.csv"

    run_folders = iter_run_folders(runs_folder)
    if not run_folders:
        raise FileNotFoundError(f"No run folders found under {runs_folder}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        for run_folder in run_folders:
            result = load_run_result(run_folder)
            if result is None:
                # Try to construct from metadata
                metadata = load_run_metadata(run_folder)
                if not metadata:
                    continue
                result = RunResult(
                    metadata=RunMetadata(
                        model=metadata.get("model", ""),
                        task=metadata.get("task", ""),
                        batch_id=metadata.get("batch_id", ""),
                        run_number=metadata.get("run_number", 0),
                    ),
                    passed=False,
                    error_message="No result file found",
                )

            row = {
                "batch_id": result.metadata.batch_id,
                "run_number": result.metadata.run_number,
                "model": result.metadata.model,
                "task": result.metadata.task,
                "timestamp": result.metadata.timestamp.isoformat(),
                "passed": result.passed,
                "assertions_passed": result.assertions_passed,
                "assertions_total": result.assertions_total,
                # Token metrics
                "input_tokens": result.stats.input_tokens,
                "output_tokens": result.stats.output_tokens,
                "total_tokens": result.stats.total_tokens,
                # Timing metrics
                "conversation_span_ms": result.stats.conversation_span_ms,
                "llm_time_ms": result.stats.llm_time_ms,
                "tool_time_ms": result.stats.tool_time_ms,
                # Conversation metrics
                "turns": result.stats.turns,
                "tool_calls": result.stats.tool_calls,
                "tool_errors": result.stats.tool_errors,
                # Tool categorization
                "mcp_calls": result.stats.mcp_calls,
                "execute_calls": result.stats.execute_calls,
                # Error and session info
                "error_message": result.error_message or "",
                "session_id": result.session_id or "",
                "session_history_file": result.session_history_file or "",
            }
            writer.writerow(row)

    return output_path


def write_batch_summary(batch_folder: Path, summary: BatchSummary) -> None:
    """Write batch summary to JSON file."""
    payload = summary.model_dump(mode="json")
    path = batch_folder / "batch_summary.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_batch_summary(batch_folder: Path) -> BatchSummary | None:
    """Load batch summary from JSON file."""
    summary_path = batch_folder / "batch_summary.json"
    if not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        return BatchSummary(**data)
    except (json.JSONDecodeError, ValueError):
        return None
