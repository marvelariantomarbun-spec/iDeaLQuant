---
description: End of session checklist - update docs and commit
---

# Session End Checklist

// turbo-all

Before ending any coding session, complete the following steps:

## 1. Update Documentation

### DEVLOG.md
Add a new entry with today's date including:
- What was accomplished
- Key decisions made
- Current status and next steps

### ROADMAP.md
Update if any milestones were completed:
- Mark completed items with ✅
- Update status of active phases
- Add any new tasks discovered

### task.md (artifact)
Update the artifact task list with:
- Completed items marked [x]
- New items discovered

## 2. Git Operations

```powershell
# Check status
git status

# Stage all changes
git add -A

# Commit with descriptive message
git commit -m "feat/fix/docs: <description>"
```

## 3. Final Verification

```powershell
# Verify clean working tree
git status
# Should show: "nothing to commit, working tree clean"
```

## Quick Command Sequence

```powershell
git add -A
git status
git commit -m "<type>: <message>"
```

## Common Commit Types
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `refactor:` - Code restructuring
- `test:` - Adding tests
