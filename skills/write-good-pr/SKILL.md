---
name: write-good-pr
description: Create clear, comprehensive pull requests that facilitate effective code review.
---

# Writing Effective Pull Requests

## Overview

A well-crafted pull request (PR) is essential for code review efficiency and project maintainability. It should clearly communicate what changed, why it changed, and how to test it.

## PR Structure

### Title Format
```
<type>: Brief description (50 chars max)
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Description Template
```markdown
## What
Brief summary of changes

## Why
Motivation and context

## How
Implementation details (if complex)

## Testing
How to test these changes

## Screenshots
(if UI changes)

## Related
Closes #123
```

## Examples

### Feature PR
**Title:** `feat: add dark mode toggle to settings`

**Description:**
```markdown
## What
Added a dark mode toggle switch in user settings

## Why
Users requested dark mode for better night-time usability (#456)

## How
- Added `darkMode` to user preferences schema
- Created toggle component with smooth transitions
- Implemented localStorage persistence

## Testing
1. Go to Settings > Appearance
2. Toggle dark mode switch
3. Refresh page - preference should persist

Closes #456
```

### Bug Fix PR
**Title:** `fix: prevent duplicate form submissions`

**Description:**
```markdown
## What
Disable submit button after click to prevent duplicates

## Why
Users accidentally submit forms multiple times, creating duplicate records

## How
- Added `isSubmitting` state to form component
- Disable button during submission
- Re-enable on error or after 3 second timeout

## Testing
1. Navigate to contact form
2. Fill required fields
3. Click submit multiple times quickly
4. Verify only one request sent

Fixes #789
```

## Best Practices

- Keep PRs small and focused (under 400 lines when possible)
- Link related issues with `Closes #123` or `Relates to #456`
- Add reviewers who understand the affected code
- Update documentation if behavior changes
- Ensure CI passes before requesting review