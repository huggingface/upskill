#!/usr/bin/env bash
#
# propose-skills.sh - Analyze Claude Code history and generate skills
#
# Usage:
#   ./propose-skills.sh              # Interactive mode - uses history + prompts
#   ./propose-skills.sh --auto       # Auto mode - uses history, generates all proposed skills
#   ./propose-skills.sh "description" # Manual mode - use provided description
#
# This script:
#   1. Extracts Claude Code conversation history
#   2. Uses Claude to analyze patterns and propose skills
#   3. Lets user select which skills to generate (or auto-selects all with --auto)
#   4. Generates properly formatted SKILL.md files
#

set -euo pipefail

# Parse arguments
AUTO_MODE=false
MANUAL_INPUT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --auto|-a)
            AUTO_MODE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--auto] [description]"
            echo "  --auto, -a    Auto-select all proposed skills"
            echo "  description   Use this description instead of history"
            exit 0
            ;;
        *)
            MANUAL_INPUT="$1"
            shift
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
CLAUDE_DIR="${CLAUDE_DIR:-$HOME/.claude}"
SKILLS_OUTPUT_DIR="${SKILLS_OUTPUT_DIR:-$HOME/.claude/skills}"
MAX_HISTORY_ITEMS="${MAX_HISTORY_ITEMS:-50}"
TEMP_DIR=$(mktemp -d)

# Cleanup on exit
trap 'rm -rf "$TEMP_DIR"' EXIT

#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1" >&2
}

#------------------------------------------------------------------------------
# Check dependencies
#------------------------------------------------------------------------------

check_dependencies() {
    local missing=()

    if ! command -v claude &> /dev/null; then
        missing+=("claude (Claude Code CLI)")
    fi

    if ! command -v jq &> /dev/null; then
        missing+=("jq")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required dependencies:"
        for dep in "${missing[@]}"; do
            echo "  - $dep"
        done
        echo ""
        echo "Install with:"
        echo "  brew install jq"
        echo "  npm install -g @anthropic-ai/claude-code"
        exit 1
    fi
}

#------------------------------------------------------------------------------
# Get Claude Code history
#------------------------------------------------------------------------------

get_history() {
    log_info "Scanning Claude Code conversation history..."

    local history_file="$TEMP_DIR/history.txt"
    local projects_dir="$CLAUDE_DIR/projects"

    if [[ ! -d "$projects_dir" ]]; then
        log_warn "No projects directory found at $projects_dir"
        return 1
    fi

    # Find recent JSONL files and extract user messages efficiently
    # Only process files modified in the last 30 days, limit to 30 files
    local raw_messages="$TEMP_DIR/raw_messages.txt"

    find "$projects_dir" -name "*.jsonl" -type f -size +0 -mtime -30 2>/dev/null | \
    head -30 | \
    xargs -I {} jq -r 'select(.type == "user") | .message.content | if type == "string" then . else empty end' {} 2>/dev/null > "$raw_messages" || true

    # Filter to command-like messages
    grep -Ei '^(help|create|make|build|add|fix|update|write|generate|run|test|check|review|explain|show|find|search|implement|refactor|debug|deploy|install|setup|configure|convert|parse|analyze|optimize|migrate|delete|remove|rename|move|copy|merge|split|format|lint|commit|push|pull|fetch|clone|init|can you|could you|please|i need|i want|how do|what is|where is|let)' "$raw_messages" 2>/dev/null | \
    grep -v '```' | \
    grep -v 'http' | \
    awk 'length >= 10 && length <= 300' | \
    sort -u | \
    head -"$MAX_HISTORY_ITEMS" > "$history_file"

    if [[ -s "$history_file" ]]; then
        local count
        count=$(wc -l < "$history_file" | tr -d ' ')
        log_success "Found $count unique user messages from history"
        cat "$history_file"
        return 0
    fi

    log_warn "No suitable history found in $projects_dir"
    return 1
}

#------------------------------------------------------------------------------
# Propose skills based on history
#------------------------------------------------------------------------------

propose_skills() {
    local history_file="$1"
    local proposals_file="$TEMP_DIR/proposals.json"
    local prompt_file="$TEMP_DIR/prompt.txt"
    local response_file="$TEMP_DIR/response.json"

    log_info "Analyzing history to propose skills..." >&2

    # Write the prompt to a file
    cat > "$prompt_file" <<'PROMPT_HEADER'
Analyze the following Claude Code conversation history and identify 3-5 recurring patterns that would benefit from being turned into reusable skills.

For each proposed skill, output a JSON object with these fields:
- name: lowercase with hyphens only (e.g., "git-workflow", "test-runner")
- description: 1-2 sentences describing what the skill does and when to trigger it
- rationale: Why this would be useful based on the history patterns
- key_instructions: Array of 3-5 strings describing what the skill should do

CRITICAL: Output ONLY a valid JSON array. No markdown code fences, no explanation, JUST the raw JSON array starting with [ and ending with ].

CONVERSATION HISTORY:
PROMPT_HEADER

    # Append the history
    cat "$history_file" >> "$prompt_file"

    # Add footer
    echo "" >> "$prompt_file"
    echo "JSON ARRAY OUTPUT:" >> "$prompt_file"

    # Call Claude using the file
    local claude_stderr="$TEMP_DIR/claude_stderr.txt"
    if ! claude -p --output-format json < "$prompt_file" > "$response_file" 2>"$claude_stderr"; then
        log_error "Failed to get response from Claude"
        cat "$claude_stderr" >&2
        return 1
    fi

    # Extract the result field
    local result
    result=$(jq -r '.result // empty' "$response_file" 2>/dev/null)

    if [[ -z "$result" ]]; then
        log_error "Empty result from Claude"
        cat "$response_file" >&2
        return 1
    fi

    # Remove markdown code fences if present
    result=$(echo "$result" | sed 's/^```json//' | sed 's/^```//' | sed 's/```$//')

    # Try to parse directly as JSON first
    if echo "$result" | jq -e 'type == "array" and length > 0' > /dev/null 2>&1; then
        echo "$result" | jq '.' > "$proposals_file"
        cat "$proposals_file"
        return 0
    fi

    # If direct parse fails, try to extract JSON array using Python (more reliable than grep for multiline)
    local json_array
    json_array=$(python3 -c "
import re, sys, json
text = sys.stdin.read()
# Find JSON array pattern
match = re.search(r'\[[\s\S]*\]', text)
if match:
    try:
        arr = json.loads(match.group())
        if isinstance(arr, list) and len(arr) > 0:
            print(json.dumps(arr))
            sys.exit(0)
    except: pass
sys.exit(1)
" <<< "$result" 2>/dev/null)

    if [[ -n "$json_array" ]] && echo "$json_array" | jq -e 'type == "array"' > /dev/null 2>&1; then
        echo "$json_array" | jq '.' > "$proposals_file"
        cat "$proposals_file"
        return 0
    fi

    log_error "Could not extract valid JSON array"
    echo "Result was: $result" | head -20 >&2
    return 1
}

#------------------------------------------------------------------------------
# Interactive skill selection
#------------------------------------------------------------------------------

select_skills() {
    local proposals_json="$1"

    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}                    PROPOSED SKILLS                            ${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""

    # Display proposals with numbers
    local count
    count=$(echo "$proposals_json" | jq 'length')

    for i in $(seq 0 $((count - 1))); do
        local name desc rationale
        name=$(echo "$proposals_json" | jq -r ".[$i].name")
        desc=$(echo "$proposals_json" | jq -r ".[$i].description")
        rationale=$(echo "$proposals_json" | jq -r ".[$i].rationale")

        echo -e "${GREEN}[$((i + 1))]${NC} ${YELLOW}$name${NC}"
        echo "    $desc"
        echo -e "    ${BLUE}Why:${NC} $rationale"
        echo ""
    done

    echo -e "${CYAN}───────────────────────────────────────────────────────────────${NC}"
    echo ""
    echo "Enter skill numbers to generate (comma-separated, e.g., '1,3,4')"
    echo "Or 'all' for all skills, 'none' to cancel:"
    echo ""
    read -r -p "> " selection

    case "$selection" in
        all|ALL|a|A)
            echo "$proposals_json"
            ;;
        none|NONE|n|N|"")
            echo "[]"
            ;;
        *)
            # Parse comma-separated numbers
            local indices=()
            IFS=',' read -ra nums <<< "$selection"
            for num in "${nums[@]}"; do
                num=$(echo "$num" | tr -d ' ')
                if [[ "$num" =~ ^[0-9]+$ ]] && [[ "$num" -ge 1 ]] && [[ "$num" -le "$count" ]]; then
                    indices+=($((num - 1)))
                fi
            done

            if [[ ${#indices[@]} -eq 0 ]]; then
                log_warn "No valid selections"
                echo "[]"
            else
                # Build selected JSON array using jq
                local jq_filter="["
                local first=true
                for idx in "${indices[@]}"; do
                    if [[ "$first" == "true" ]]; then
                        first=false
                    else
                        jq_filter+=","
                    fi
                    jq_filter+=".[$idx]"
                done
                jq_filter+="]"
                echo "$proposals_json" | jq "$jq_filter"
            fi
            ;;
    esac
}

#------------------------------------------------------------------------------
# Generate SKILL.md file
#------------------------------------------------------------------------------

generate_skill() {
    local skill_json="$1"
    local output_dir="$2"

    local name desc
    name=$(echo "$skill_json" | jq -r '.name')
    desc=$(echo "$skill_json" | jq -r '.description')

    # Format key instructions as markdown list
    local instructions
    instructions=$(echo "$skill_json" | jq -r '.key_instructions[]' 2>/dev/null | sed 's/^/- /' || echo "- Follow the workflow")

    # Create skill directory
    local skill_dir="$output_dir/$name"
    mkdir -p "$skill_dir"

    log_info "Generating SKILL.md for: $name"

    local prompt_file="$TEMP_DIR/skill_prompt.txt"
    local response_file="$TEMP_DIR/skill_response.txt"

    # Write skill generation prompt
    cat > "$prompt_file" <<PROMPT_END
Generate a complete SKILL.md file for a Claude Code skill.

Skill Details:
- Name: $name
- Description: $desc
- Key Instructions:
$instructions

Requirements:
1. Start with YAML frontmatter (--- delimiters) containing only name and description
2. Include sections: When to Use, Workflow, Examples, Best Practices
3. Be specific and actionable
4. Keep under 200 lines

Output ONLY the SKILL.md content. Start with --- and end after Best Practices. No other text.
PROMPT_END

    local skill_content
    if claude -p < "$prompt_file" > "$response_file" 2>/dev/null; then
        skill_content=$(cat "$response_file")

        # Clean up the content using Python for reliability
        skill_content=$(python3 -c "
import sys, re

content = sys.stdin.read()

# Remove markdown code fences
content = re.sub(r'^\`\`\`[a-z]*\n', '', content)
content = re.sub(r'\n\`\`\`\s*$', '', content)
content = content.strip()

# Find all frontmatter blocks (--- ... ---)
fm_pattern = r'^---\n(.*?)\n---'
matches = list(re.finditer(fm_pattern, content, re.MULTILINE | re.DOTALL))

if len(matches) >= 2:
    # Keep only the first frontmatter and content after the last frontmatter
    first_fm = matches[0].group(0)
    last_fm_end = matches[-1].end()
    body = content[last_fm_end:].strip()
    content = first_fm + '\n\n' + body
elif len(matches) == 1:
    # Already has exactly one frontmatter, keep as is
    pass
# else: no frontmatter, will be added later

print(content)
" <<< "$skill_content" 2>/dev/null) || skill_content=$(cat "$response_file")
    fi

    # Fallback if Claude fails or returns empty
    if [[ -z "$skill_content" ]] || [[ ${#skill_content} -lt 50 ]]; then
        log_warn "Using template for $name"
        skill_content="---
name: $name
description: $desc
---

# ${name//-/ }

## When to Use

$desc

## Workflow

$instructions

## Examples

### Example 1
\`\`\`
[Add example input/output here]
\`\`\`

## Best Practices

- Follow consistent patterns
- Verify results before completing
- Ask for clarification when requirements are ambiguous
"
    fi

    # Ensure content starts with frontmatter
    if [[ ! "$skill_content" =~ ^--- ]]; then
        skill_content="---
name: $name
description: $desc
---

$skill_content"
    fi

    # Write SKILL.md
    echo "$skill_content" > "$skill_dir/SKILL.md"

    log_success "Generated: $skill_dir/SKILL.md"
}

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------

main() {
    echo ""
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║           CLAUDE CODE SKILL PROPOSAL GENERATOR                ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Check dependencies
    log_info "Checking dependencies..."
    check_dependencies
    log_success "Dependencies OK"

    # Get history
    local history_file="$TEMP_DIR/history.txt"

    # Use manual input if provided
    if [[ -n "$MANUAL_INPUT" ]]; then
        log_info "Using provided description"
        echo "$MANUAL_INPUT" > "$history_file"
    else
        log_info "Looking for history..."
        # get_history writes to its own file and outputs to stdout
        # Capture stdout (the actual history) to the file
        if get_history > "$history_file" 2>/dev/null; then
            # Check if we got valid content
            if [[ ! -s "$history_file" ]] || ! grep -q '[a-zA-Z]' "$history_file" 2>/dev/null; then
                log_warn "History file empty or invalid"
                : > "$history_file"  # Clear the file
            fi
        else
            : > "$history_file"  # Create empty file on failure
        fi
    fi

    if [[ ! -s "$history_file" ]]; then
        # Check if running interactively
        if [[ -t 0 ]]; then
            echo ""
            log_info "No history found. Enter a description of your common Claude Code usage patterns:"
            log_info "(What tasks do you frequently ask Claude to help with?)"
            echo ""
            read -r -p "> " user_input

            if [[ -z "$user_input" ]]; then
                log_error "No history or description provided. Exiting."
                exit 1
            fi
            echo "$user_input" > "$history_file"
        else
            log_error "No history found and not running interactively."
            log_error "Run this script in an interactive terminal, or provide a description as argument."
            exit 1
        fi
    fi

    # Propose skills
    local proposals
    if ! proposals=$(propose_skills "$history_file"); then
        log_error "Failed to generate skill proposals"
        exit 1
    fi

    if [[ -z "$proposals" ]] || [[ "$proposals" == "[]" ]] || [[ "$proposals" == "null" ]]; then
        log_error "No skill proposals generated. Try providing more context."
        exit 1
    fi

    # Validate JSON
    if ! echo "$proposals" | jq -e '.' > /dev/null 2>&1; then
        log_error "Invalid proposals JSON"
        echo "Proposals: $proposals" | head -5
        exit 1
    fi

    # Select skills
    local selected
    if [[ "$AUTO_MODE" == "true" ]]; then
        log_info "Auto mode: selecting all proposed skills"
        selected="$proposals"
    else
        selected=$(select_skills "$proposals")
    fi

    if [[ "$selected" == "[]" ]]; then
        log_info "No skills selected. Exiting."
        exit 0
    fi

    # Confirm output directory
    echo ""
    log_info "Skills will be generated in: $SKILLS_OUTPUT_DIR"
    if [[ "$AUTO_MODE" != "true" ]] && [[ -t 0 ]]; then
        read -r -p "Press Enter to continue or type a different path: " custom_dir
        if [[ -n "$custom_dir" ]]; then
            SKILLS_OUTPUT_DIR="$custom_dir"
        fi
    fi

    mkdir -p "$SKILLS_OUTPUT_DIR"

    # Generate each selected skill
    echo ""
    log_info "Generating skills..."
    echo ""

    local skill_count
    skill_count=$(echo "$selected" | jq 'length')

    for i in $(seq 0 $((skill_count - 1))); do
        local skill
        skill=$(echo "$selected" | jq ".[$i]")
        generate_skill "$skill" "$SKILLS_OUTPUT_DIR"
    done

    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}                    GENERATION COMPLETE                        ${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Skills generated in: $SKILLS_OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Review and customize each SKILL.md file"
    echo "  2. Add scripts/ or references/ directories as needed"
    echo "  3. Skills in ~/.claude/skills/ are auto-discovered by Claude Code"
    echo ""

    # List generated skills
    echo "Generated skills:"
    for dir in "$SKILLS_OUTPUT_DIR"/*/; do
        if [[ -d "$dir" ]]; then
            echo "  - $(basename "$dir")"
        fi
    done
    echo ""
}

# Run main
main "$@"
