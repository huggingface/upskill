<img width="1920" height="1080" alt="upskill_banner" src="https://github.com/user-attachments/assets/b71fd417-7d23-4f5d-aa89-06ea6b284d1b" />

# UPskill

Generate and evaluate agent skills based on traces with agents. Create skills with teacher models (expensive/slow) that student models (cheap/fast) can use to perform harder tasks reliably.

## Quick Start

Install upskill:

```bash
pip install upskill
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
- `--tool` - Generate from MCP tool schema (path#tool_name)
- `-f, --from PATH` - Improve from existing skill dir or agent trace file (auto-detected)
- `-m, --model MODEL` - Model for generation (e.g., 'sonnet', 'haiku', 'anthropic.claude-sonnet-4-20250514')
- `-o, --output PATH` - Output directory for skill
- `--no-eval` - Skip evaluation and refinement
- `--eval-model MODEL` - Different model to evaluate skill on
- `--eval-provider [anthropic|openai|generic]` - API provider for eval model
- `--eval-base-url URL` - Custom API endpoint for eval model
- `--runs-dir PATH` - Directory for run logs (default: ./runs)
- `--log-runs / --no-log-runs` - Log run data (default: enabled)

**Examples:**

```bash
# Basic usage
upskill generate "parse JSON Schema files"

# Make and evaluate skills for less powerful models
upskill generate "write git commits" --model sonnet --eval-model haiku

# Improve an existing skill (auto-detected as directory)
upskill generate "add more error handling examples" --from ./skills/api-errors/

# Generate from an agent trace file (auto-detected as file)
upskill generate "document the pattern" --from ./trace.json

# Evaluate on local model (llama.cpp server)
upskill generate "parse YAML" \
    --eval-model "unsloth/GLM-4.7-Flash-GGUF:Q4_0" \
    --eval-base-url http://localhost:8080/v1
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
- `--runs N` - Number of runs per model (default: 1)
- `--provider [anthropic|openai|generic]` - API provider (auto-detected as 'generic' when --base-url is provided)
- `--base-url URL` - Custom API endpoint for local models
- `--no-baseline` - Skip baseline comparison
- `-v, --verbose` - Show per-test results
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

# Evaluate on local model (llama.cpp server)
upskill eval ./skills/my-skill/ \
    -m "unsloth/GLM-4.7-Flash-GGUF:Q4_0" \
    --base-url http://localhost:8080/v1

# Skip baseline (just test with skill)
upskill eval ./skills/my-skill/ --no-baseline

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

### `upskill delete`

Delete a generated skill safely from your skills directory.

```bash
upskill delete SKILL_NAME [OPTIONS]
```

**Arguments:**
- `SKILL_NAME` - Name of the skill directory to remove (not a path)

**Options:**
- `-d, --dir PATH` - Skills directory (default: configured `skills_dir`)
- `--force` - Skip confirmation prompt

**Examples:**

```bash
# Confirm before deleting
upskill delete git-commit-messages

# Delete without confirmation
upskill delete my-skill --force
```

### `upskill runs`

View run results as a plot, or export to CSV. By default, shows a visual comparison of baseline vs with-skill performance.

```bash
upskill runs [OPTIONS]
```

**Options:**
- `-d, --dir PATH` - Runs directory
- `-s, --skill TEXT` - Filter by skill name(s) (repeatable)
- `-m, --model TEXT` - Filter by model(s) (repeatable)
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

### upskill config (`~/.config/upskill/config.yaml`)

```yaml
model: sonnet                    # Default generation model
eval_model: haiku               # Default evaluation model (optional)
skills_dir: ./skills            # Where to save skills
runs_dir: ./runs                # Where to save run logs
max_refine_attempts: 3          # Refinement iterations
```

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

upskill supports local models through any OpenAI-compatible endpoint (Ollama, llama.cpp, vLLM, etc.).

**Quick start with Ollama:**

```bash
# Start Ollama (default port 11434)
ollama serve

# Evaluate with a local model
upskill eval ./skills/my-skill/ \
    --model llama3.2:latest \
    --base-url http://localhost:11434/v1
```

**With llama.cpp server:**

```bash
# Start llama.cpp server
./llama-server -m model.gguf --port 8080

# Evaluate with the local model
upskill eval ./skills/my-skill/ \
    --model my-model \
    --base-url http://localhost:8080/v1
```

When `--base-url` is provided, the provider is automatically set to `generic` unless you specify `--provider` explicitly.
