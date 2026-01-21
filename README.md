<img width="1920" height="1080" alt="upskill_banner" src="https://github.com/user-attachments/assets/b71fd417-7d23-4f5d-aa89-06ea6b284d1b" />

# upskill

Generate and evaluate agent skills using FastAgent. Create instruction documents that teach AI coding agents how to perform tasks reliably.

## Installation

```bash
pip install upskill
# or
uvx upskill
```

## Quick Start

```bash
# Generate a skill from <img width="1920" height="1080" alt="upskill_banner" src="https://github.com/user-attachments/assets/4c667844-b1f1-4ee6-aee9-a6abe557fc6c" />
a task description
upskill generate "write good git commit messages"

# Skills are saved to ./skills/{skill-name}/ by default
# Run logs are saved to ./runs/ by default
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
- `--trace PATH` - Generate from agent trace file
- `-m, --model MODEL` - Model for generation (e.g., 'sonnet', 'haiku', 'anthropic.claude-sonnet-4-20250514')
- `-o, --output PATH` - Output directory for skill
- `--no-eval` - Skip evaluation and refinement
- `--eval-model MODEL` - Different model to evaluate skill on
- `--eval-provider [anthropic|openai|generic]` - API provider for eval model (auto-detected as 'generic' when --eval-base-url is provided)
- `--eval-base-url URL` - Custom API endpoint for eval model
- `--runs-dir PATH` - Directory for run logs (default: ./runs)
- `--log-runs / --no-log-runs` - Log run data (default: enabled)

**Examples:**

```bash
# Basic usage
upskill generate "parse JSON Schema files"

# Generate a skill based on a trace file
upskill generate "parse JSON Schema files" --trace ./conversation_with_claude_sonnet.md

# Make and evaluate skills for less powerful models
upskill generate "write git commits" --model sonnet --eval-model haiku

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

Evaluate an existing skill against test cases.

```bash
upskill eval SKILL_PATH [OPTIONS]
```

**Arguments:**
- `SKILL_PATH` - Path to skill directory containing SKILL.md

**Options:**
- `-t, --tests PATH` - Test cases JSON file
- `-m, --model MODEL` - Model to evaluate against
- `--provider [anthropic|openai|generic]` - API provider (auto-detected as 'generic' when --base-url is provided)
- `--base-url URL` - Custom API endpoint for local models
- `--no-baseline` - Skip baseline comparison
- `-v, --verbose` - Show per-test results
- `--log-runs / --no-log-runs` - Log run data (default: enabled)
- `--runs-dir PATH` - Directory for run logs

**Examples:**

```bash
# Basic evaluation
upskill eval ./skills/my-skill/

# With verbose output
upskill eval ./skills/my-skill/ -v

# Custom test cases
upskill eval ./skills/my-skill/ --tests ./tests.json

# Evaluate on specific model
upskill eval ./skills/my-skill/ -m haiku

# Evaluate on local model (llama.cpp server)
upskill eval ./skills/my-skill/ \
    --model "unsloth/GLM-4.7-Flash-GGUF:Q4_0" \
    --base-url http://localhost:8080/v1

# Skip baseline (just test with skill)
upskill eval ./skills/my-skill/ --no-baseline

# Disable run logging
upskill eval ./skills/my-skill/ --no-log-runs
```

**Test cases JSON format:**

```json
[
  {"input": "Write a commit for adding login", "expected": {"contains": "feat"}},
  {"input": "Fix the null pointer bug", "expected": {"contains": "fix"}}
]
```

### `upskill list`

List all generated skills.

```bash
upskill list [OPTIONS]
```

**Options:**
- `-d, --dir PATH` - Skills directory to list

**Examples:**

```bash
# List skills in default directory
upskill list

# List from custom directory
upskill list -d ./my-skills/
```

### `upskill runs`

Summarize run logs to CSV.

```bash
upskill runs [OPTIONS]
```

**Options:**
- `-d, --dir PATH` - Runs directory
- `--csv PATH` - Output CSV path

**Examples:**

```bash
# Summarize default runs directory
upskill runs

# Custom runs directory
upskill runs -d ./my-runs/

# Custom output path
upskill runs --csv ./results.csv
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
