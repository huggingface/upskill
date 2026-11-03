<img width="1920" height="1080" alt="upskill_banner" src="https://github.com/user-attachments/assets/b71fd417-7d23-4f5d-aa89-06ea6b284d1b" />

# UPskill

Generate and evaluate agent skills based on traces with agents. Create skills with teacher models (expensive/slow) that student models (cheap/fast) can use to perform harder tasks reliably.


> [!TIP] 
> 
> UPskill v2 - recommended default config file now runs evaluations on Hugging Face Jobs. Make sure
> to set your `HF_TOKEN` and use `--artifact-repo <dataset-name>` for job creation and result capture  

## Quick Start

Install upskill:

```bash
uv pip install upskill
# or just use uv
uvx upskill
```

Create a new skill

```bash
upskill generate "write good git commit messages"
# or based on previous agent traces
upskill generate "document the pattern" --from ./trace.md
# Skills are saved to ./skills/{skill-name}/ by default
```

Generate a skill with a teaching model and evaluate it on a student model.

```bash
upskill generate "write good git commit messages" --model sonnet --eval-model haiku
```

Benchmark a set of models against a skill.

```bash
upskill eval ./skills/git-commit-messages/ -m haiku -m sonnet
# logs pretty printed to the terminal
```

View the results later.

```bash
upskill runs --skill git-commit-messages
```

## Development checks

This repo uses a CI flow inspired by `fast-agent` with separate format, lint, typecheck, and test
stages.

Install dev dependencies:

```bash
uv sync --extra dev
```

Run the quality gates locally:

```bash
uv run scripts/format.py
uv run scripts/lint.py
uv run scripts/typecheck.py
uv run scripts/cpd.py --check
uv run --extra dev pytest -v
```

Or use the helper script to run the whole sequence:

```bash
uv run scripts/check.py
```

Add `--sync` to include `uv sync --extra dev`, or `--skip-tests` for a faster static-only pass.

To auto-format before re-running checks:

```bash
uv run --extra dev scripts/format.py --write
```

Current enforced standards:

- `ruff format --check` for formatting
- `ruff check` for style, imports, modernization, bugbear, simplify, and import-hygiene rules
- cyclomatic complexity via Ruff `C90` with `max-complexity = 15`
- `ty check` across `src`, `tests`, and `scripts`
- `pmd cpd` via `scripts/cpd.py --check` to flag duplicated code in `src/`
- `pytest` for the test suite

CI enforcement lives in `.github/workflows/ci.yml` and runs on pushes and pull requests targeting
`main`.

## Model Handling Overview

upskill uses distinct phases with explicit model roles:

- **Skill generation**: create/refine `SKILL.md`
- **Test generation**: create synthetic evaluation cases
- **Evaluation**: run tests against evaluator model(s)
- **Benchmark**: repeated evaluation across multiple runs/models

Model flags by command:

| Command | Flag | Meaning |
|---|---|---|
| `generate` | `--model` | Skill generation/refinement model |
| `generate` | `--test-gen-model` | Test generation model override |
| `generate` | `--eval-model` | Optional extra cross-model eval pass |
| `eval` | `-m/--model` | Evaluation model(s) (repeatable) |
| `eval` | `--test-gen-model` | Test generation model override (when tests are generated) |
| `benchmark` | `-m/--model` | Evaluation model(s) to benchmark |
| `benchmark` | `--test-gen-model` | Test generation model override (when tests are generated) |
| `runs` / `plot` | `-m/--model` | Historical results filter only |

`upskill eval` enters **benchmark mode** whenever you pass multiple `-m` values or `--runs > 1`.
In benchmark mode, baseline comparison is always off; `--no-baseline` is redundant.

If a verifier raises an exception (for example, an unreadable output file or a
command timeout), evaluation records an error for that test case and continues
with the remaining cases and baseline phase. The per-test `test_result.json`
preserves the agent output, usage statistics, and exception type and message.
An ordinary failed assertion still has a structured `validation_result`;
interruption and cancellation are not converted into test failures.

## Commands

### `upskill generate`

Generate a skill from a task description with automatic evaluation and refinement.

```bash
upskill generate TASK [OPTIONS]
```

**Arguments:**
- `TASK` - Description of what the skill should teach

**Options:**
- `-e, --example` - Input -> output example (can be repeated)
- `-f, --from PATH` - Improve from existing skill dir or agent trace file (auto-detected)
- `-m, --model MODEL` - Skill generation model (e.g., 'sonnet', 'haiku', 'anthropic.claude-sonnet-4-20250514')
- `--test-gen-model MODEL` - Override test generation model for this run
- `-o, --output PATH` - Output directory for skill
- `--no-eval` - Skip evaluation and refinement
- `--eval-model MODEL` - Different model to evaluate skill on
- `--executor [local|jobs]` - Execution backend for evaluation/refinement; overrides config
- `--artifact-repo TEXT` - Dataset repo for remote fast-agent job artifacts (required with `--executor jobs`)
- `--max-parallel N` - Max concurrent evaluation executions; overrides config
- `--runs-dir PATH` - Directory for run logs (default: ./runs)
- `--log-runs / --no-log-runs` - Log run data (default: enabled)

**Examples:**

```bash
# Basic usage
upskill generate "parse JSON Schema files"

# Make and evaluate skills for less powerful models
upskill generate "write git commits" --model sonnet --eval-model haiku

# Remote execution on Hugging Face Jobs
upskill generate "parse invoices" --executor jobs --artifact-repo <user>/upskill-tests

# Improve an existing skill (auto-detected as directory)
upskill generate "add more error handling examples" --from ./skills/api-errors/

# Generate from an agent trace file (auto-detected as file)
upskill generate "document the pattern" --from ./trace.json

# Skip evaluation during generation (evaluate separately with upskill eval)
upskill generate "parse YAML" --no-eval
```

**Output:**

```
Generating skill with sonnet...
Generating test cases...
Evaluating on sonnet... (attempt 1)
  60% -> 100% (+40%) OK

  git-commit-messages
  Write clear, conventional commit messages that follow best practices.

  SKILL.md              ~450 tokens

  baseline   ████████████░░░░░░░░   60%
  with skill ████████████████████  100%  (+40%)

  tokens: 1200 → 800  (-33%)

Saved to ./skills/git-commit-messages
```

### `upskill eval`

Evaluate an existing skill against test cases. Supports single-model evaluation with baseline comparison, or multi-model benchmarking.

```bash
upskill eval SKILL_PATH [OPTIONS]
```

**Arguments:**
- `SKILL_PATH` - Path to skill directory containing SKILL.md

**Options:**
- `-t, --tests PATH` - Test cases JSON file
- `-m, --model MODEL` - Model(s) to evaluate against (repeatable for multi-model benchmarking)
- `--test-gen-model MODEL` - Override test generation model when tests must be generated
- `--runs N` - Number of runs per model; overrides config
- `--no-baseline` - Skip baseline comparison (simple eval mode only; ignored in benchmark mode)
- `-v, --verbose` - Show per-test results
- `--executor [local|jobs]` - Execution backend for evaluation; overrides config
- `--max-parallel N` - Max concurrent evaluation executions; overrides config
- `--log-runs / --no-log-runs` - Log run data (default: enabled)
- `--runs-dir PATH` - Directory for run logs

**Examples:**

```bash
# Basic evaluation with baseline comparison
upskill eval ./skills/my-skill/

# With verbose output
upskill eval ./skills/my-skill/ -v

# Custom test cases
upskill eval ./skills/my-skill/ --tests ./tests.json

# Evaluate on specific model
upskill eval ./skills/my-skill/ -m haiku

# Multi-model benchmarking (compare models)
upskill eval ./skills/my-skill/ -m haiku -m sonnet

# Multiple runs per model for statistical significance
upskill eval ./skills/my-skill/ -m haiku -m sonnet --runs 5

# Evaluate a local model configured in fast-agent
upskill eval ./skills/my-skill/ -m generic.my-model

# Skip baseline (just test with skill)
upskill eval ./skills/my-skill/ --no-baseline

# Benchmark mode is triggered by multiple models OR --runs > 1
upskill eval ./skills/my-skill/ -m haiku --runs 5

# Disable run logging
upskill eval ./skills/my-skill/ --no-log-runs
```

**Benchmark output:**

```
Evaluating my-skill across 2 model(s)
  3 test case(s), 5 run(s) per model

haiku
  Pass rate: 4/5 (80%)  Avg assertions: 2.8/3

sonnet
  Pass rate: 5/5 (100%)  Avg assertions: 3.0/3

┏━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Model  ┃ Pass Rate ┃ Avg Assertions ┃ Avg Tokens ┃
┡━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ haiku  │ 4/5       │ 2.8/3          │ 1250       │
│ sonnet │ 5/5       │ 3.0/3          │ 1890       │
└────────┴───────────┴────────────────┴────────────┘
```

**Test cases JSON format:**

```json
[
  {"input": "Write a commit for adding login", "expected": {"contains": ["feat", "login"]}},
  {"input": "Fix the null pointer bug", "expected": {"contains": ["fix", "bug"]}}
]
```

### `upskill list`

List all generated skills in a tree view.

```bash
upskill list [OPTIONS]
```

**Options:**
- `-d, --dir PATH` - Skills directory to list
- `-v, --verbose` - Show skill contents preview

**Examples:**

```bash
# List skills in default directory
upskill list

# List from custom directory
upskill list -d ./my-skills/

# Show preview of skill contents
upskill list -v
```

**Output:**

```
./skills
├── git-commit-messages
│   ├── Write clear, conventional commit messages...
│   └── files
│       └── SKILL.md
├── api-error-handling
│   ├── Handle API errors gracefully with proper logging...
│   └── files
│       ├── SKILL.md
│       └── references/error-codes.md
└── yaml-parsing
    ├── Parse YAML files safely with schema validation...
    └── files
        ├── SKILL.md
        └── scripts/validate.py
```

### `upskill runs`

View run results as a plot, or export to CSV. By default, shows a visual comparison of baseline vs with-skill performance.

```bash
upskill runs [OPTIONS]
```

**Options:**
- `-d, --dir PATH` - Runs directory
- `-s, --skill TEXT` - Filter by skill name(s) (repeatable)
- `-m, --model TEXT` - Filter historical run data by model(s) (repeatable)
- `--metric [success|tokens]` - Metric to display (default: success)
- `--csv PATH` - Export to CSV instead of plot

**Examples:**

```bash
# View results plot (default)
upskill runs

# Filter by skill and models
upskill runs -s my-skill -m haiku -m sonnet

# Show token usage instead of success rate
upskill runs --metric tokens

# Export to CSV
upskill runs --csv ./results.csv

# Custom runs directory
upskill runs -d ./my-runs/
```

**Plot output:**

```
skill: git-commit-messages

haiku
  baseline   ████████████░░░░░░░░   60%
  with skill ████████████████░░░░   80%  (+20%)

sonnet
  baseline   ████████████░░░░░░░░   60%
  with skill ████████████████████  100%  (+40%)
```

**Matrix view (multiple skills and models):**

```
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ skill               ┃ haiku        ┃ sonnet       ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ git-commit-messages │ 60%→80%      │ 60%→100%     │
│ api-error-handling  │ 40%→70%      │ 50%→90%      │
│ yaml-parsing        │ 70%→90%      │ 80%→100%     │
└─────────────────────┴──────────────┴──────────────┘
```

## Skill Output Format

Skills are saved in a standard directory format:

```
./skills/{skill-name}/
├── SKILL.md          # Main skill instructions
├── references/       # Supporting documents (optional)
└── scripts/          # Executable scripts (optional)
```

**Example SKILL.md:**

```markdown
# git-commit-messages

Write clear, conventional commit messages that follow best practices.

## Instructions

This skill teaches how to write effective git commit messages
following the Conventional Commits specification.

## Format

Commit messages should follow this structure:

<type>(<scope>): <subject>

<body>

<footer>

## Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
...

## Examples

### Simple feature commit
feat(auth): add password reset functionality

### Bug fix with explanation
fix(api): handle null response from user service

The user service can return null when not found.
Added proper null checking to prevent crashes.

Closes #123
```

## Run Logging

By default, upskill logs all runs to `./runs/`. Each run creates:

```
./runs/
├── 2025_01_21_15_30/           # Batch folder (timestamp)
│   ├── run_1/
│   │   ├── run_metadata.json   # Model, task, timing
│   │   └── run_result.json     # Pass/fail, assertions, tokens
│   ├── run_2/
│   │   └── ...
│   └── batch_summary.json      # Aggregate results
└── results.csv                 # Summary CSV (after `upskill runs`)
```

Disable with `--no-log-runs`.

## Configuration

### upskill config (`./upskill.config.yaml`)

```yaml
skill_generation_model: sonnet   # Default skill generation model
eval_model: haiku               # Default evaluation model (optional)
test_gen_model: null            # Optional test generation model
skills_dir: ./skills            # Where to save skills
runs_dir: ./runs                # Where to save run logs
max_refine_attempts: 2          # Refinement iterations
executor: local                 # Default execution backend
num_runs: 1                     # Default eval/benchmark runs when --runs is omitted
max_parallel: 5                 # Default concurrent evaluation executions
jobs_secrets: HF_TOKEN          # Comma-separated HF Jobs env var names to forward
jobs_image: ghcr.io/astral-sh/uv:python3.13-bookworm  # HF Jobs container image
# fastagent_config: ./fastagent.config.yaml  # Optional FastAgent config override
```

`test_gen_model` fallback behavior:

- CLI `--test-gen-model` overrides config for a single run.
- If set, test generation uses `test_gen_model`.
- If unset, test generation falls back to `skill_generation_model`.
- For `eval`/`benchmark`, this intentionally uses `skill_generation_model` (not `eval_model`) so generated tests stay
  stable when sweeping multiple evaluation models.

Backward compatibility: `model` is still accepted in config files as a legacy alias for
`skill_generation_model`.

CLI flags override config values for execution settings:

- `--executor` overrides `executor`
- `--runs` overrides `num_runs`
- `--max-parallel` overrides `max_parallel`
- `--jobs-secrets` overrides `jobs_secrets`

If you set `executor: jobs`, you still need the required jobs-specific CLI inputs such as
`--artifact-repo`.

`jobs_secrets` is a comma-separated list of environment variable names to forward into
remote HF Jobs runs. It should contain secret names such as `HF_TOKEN` or
`ANTHROPIC_API_KEY`, not literal secret values.

`jobs_image` controls which container image HF Jobs uses for remote execution.

Config lookup order:

1. `UPSKILL_CONFIG` environment variable (path)
2. `./upskill.config.yaml` (project local)
3. `~/.config/upskill/config.yaml` (legacy fallback)

### FastAgent config (`fastagent.config.yaml`)

Place in your project directory to customize FastAgent settings:

```yaml
default_model: sonnet

logger:
  progress_display: true
  show_chat: false
  streaming: markdown

# MCP servers (optional)
mcp:
  servers:
    fetch:
      command: "uvx"
      args: ["mcp-server-fetch"]
```

## Environment Variables

```bash
# Required for Anthropic models
ANTHROPIC_API_KEY=sk-ant-...

# Required for OpenAI models
OPENAI_API_KEY=sk-...

# Optional: custom endpoints
ANTHROPIC_BASE_URL=http://localhost:8080
OPENAI_API_BASE=http://localhost:11434/v1

# For local models (generic provider)
GENERIC_BASE_URL=http://localhost:8080/v1
GENERIC_API_KEY=local  # Optional, defaults to "local"
```

## Python API

```python
from upskill import (
    generate_skill,
    generate_tests,
    evaluate_skill,
    refine_skill,
    Config,
)

# Load configuration
config = Config.load()

# Generate a skill
skill = await generate_skill(
    "parse JSON Schema files",
    model="sonnet",
    config=config,
)

# Generate test cases
tests = await generate_tests("parse JSON Schema files")

# Evaluate the skill
results = await evaluate_skill(
    skill,
    tests,
    model="haiku",
    config=config,
)

print(f"Skill lift: {results.skill_lift:.0%}")
print(f"Token savings: {results.token_savings:.0%}")
print(f"Is beneficial: {results.is_beneficial}")

# Refine based on failures
if not results.is_beneficial:
    from upskill.evaluate import get_failure_descriptions
    failures = get_failure_descriptions(results)
    improved_skill = await refine_skill(skill, failures)
```

## Model Format

upskill uses FastAgent model format:

```
<provider>.<model>.<reasoning_effort?>
```

**Examples:**
- `sonnet` - Anthropic Claude Sonnet (alias)
- `haiku` - Anthropic Claude Haiku (alias)
- `opus` - Anthropic Claude Opus (alias)
- `anthropic.claude-sonnet-4-20250514` - Full model name
- `openai.gpt-4.1` - OpenAI GPT-4.1
- `openai.o3-mini.low` - OpenAI o3-mini with low reasoning effort
- `generic.llama3.2:latest` - Local model via Ollama
- `generic.my-model` - Local model via llama.cpp or other OpenAI-compatible server

## Local Models

upskill supports local models through any OpenAI-compatible endpoint (llama.cpp, vLLM, etc.).

**With llama.cpp server:**

```bash
# Start llama.cpp server
./llama-server -m model.gguf --port 8080

# Configure endpoint via fast-agent config/env, then evaluate
upskill eval ./skills/my-skill/ --model generic.my-model
```
