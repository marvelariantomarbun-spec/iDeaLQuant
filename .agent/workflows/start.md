---
description: Start session checklist - project status report
---

# Session Start Protocol

// turbo-all

When restarting a session or typing `/start`, follow these steps to generate a "Welcome Back" report:

## 1. Analyze Project State

```python
# Read core documentation
read_file("task.md")      # For active/pending tasks
read_file("ROADMAP.md")   # For high-level phase status
read_file("DEVLOG.md")    # For recent history (last entry)
```

## 2. Check Environment

```powershell
# Check git status
git status
```

## 3. Generate Status Report

Produce a concise report including:

### 🚀 Status Summary
- **Current Phase:** (From ROADMAP.md)
- **Active Task:** (From task.md)
- **Last Achievement:** (From DEVLOG.md)

### 📋 Next Steps
- List the top 2-3 pending items from `task.md`.

### ❓ Decision
- Propose the immediate next action.
- Ask user for confirmation or if there's a new priority.
