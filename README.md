# upskill

Generate agent skills based on traces, MCP tools, or example input/output pairs.

## Usage

Use your Claude Code export or MCP tool schema to generate a skill that improves your agent's performance.

```bash
uvx upskill generate "skill to write tests for new features" \
      --trace <claude-code-export.md> \
      --tool <mcp-tool.json>
```

Take your new skill and use it on a different (cheaper|local) model and see how it performs.

```bash
uvx upskill generate "skill to document code in our style guide" \
    --model claude-sonnet-4-20250514 \
    --trace <claude-code-export.md> \
    --eval-model unsloth/GLM-4.7-Flash-GGUF:Q4_0 \
    --eval-base-url http://127.0.0.1:8080
```

Output:

```
                      Skill Impact by Model
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━┓
┃ model                           ┃ baseline ┃ with skill ┃ lift ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━┩
│ claude-sonnet-4-20250514        │ 40%      │ 100%       │ +60% │
│ unsloth/GLM-4.7-Flash-GGUF:Q4_0 │ 60%      │ 100%       │ +40% │
└─────────────────────────────────┴──────────┴────────────┴──────┘
```

### Evaluate existing skill

```bash
uvx upskill eval ./my-skill/ -m unsloth/GLM-4.7-Flash-GGUF:Q4_0 --base-url http://127.0.0.1:8080
```

## Environment

Set `ANTHROPIC_API_KEY` in `.env` or environment.
