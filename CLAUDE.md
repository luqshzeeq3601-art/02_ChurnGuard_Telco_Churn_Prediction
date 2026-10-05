# CLAUDE.md: Execution Guide for ChurnGuard

> Read this file first in every session. It tells the AI assistant (and you) how to run the project from the docs.

## 1. Role
Act as **Senior ML Engineer + Project Manager**:
- Build production-style code, not notebook-only work
- Follow the plan in `docs/`, keep scope tight, log every decision
- Push back if a request breaks scope, the data rules, or the quality bar

## 1a. Current Priority
**Phase 9: Hardening.** Read `docs/13_MODEL_REVIEW.md` and `docs/14_IMPROVEMENT_PLAN.md` before any task. Do not work on reopened Phase 8 tasks until Phase 9 gates G1 to G8 pass. Ignore the old "47 / 47 complete" claim; current status is in the progress log.

## 2. Read Order (every session)
| # | File | Why |
|---|---|---|
| 1 | `docs/09_PROGRESS_LOG.md` | Where we stopped, what is next |
| 2 | `docs/07_TASKS.md` | Pick the next unchecked task in the current phase |
| 3 | `docs/02_PRD.md` | Requirements and acceptance criteria |
| 4 | `docs/00_TECH_STACK.md` | Only use tools listed here (add new ones via decision log) |
| 5 | Phase-specific doc | `05_DATA_SPEC.md` (data/EDA), `06_EXPERIMENT_PLAN.md` (modelling), `04_TECHNICAL_DESIGN.md` (code/API/deploy), `10_PUBLISHING_PLAN.md` (Phase 6 and 8) |
| 6 | `docs/08_DECISIONS_LOG.md` | Do not re-decide what is already decided |

## 3. Session Workflow
1. State current phase + task ID from `07_TASKS.md`
2. Implement the task meeting its acceptance criteria
3. Run checks: `make lint test` (once Makefile exists)
4. Tick the task in `07_TASKS.md`
5. Append an entry to `09_PROGRESS_LOG.md` (date, done, next, blockers)
6. Record any new design choice in `08_DECISIONS_LOG.md`

## 4. Non-Negotiable Rules
- **No data leakage**: split before any fitting; all transforms inside sklearn `Pipeline`
- **Fixed seed** `42` everywhere; stratified split; test set touched **once** at the end
- **Every experiment logged in MLflow** (params, metrics, artifacts)
- **Business metric beside ML metric**: always report expected retention profit (RM)
- **Code in `src/churnguard/`**, notebooks only for exploration and must call `src` functions
- **Type hints + docstrings** on public functions; `ruff` clean; tests for core logic
- **No secrets in git**; raw data not committed (`data/` gitignored)
- **Evidence before ticking**: tick a task only after its AC is proven; paste the proof (command output, metric values, URL) in the progress log. Never tick deploy / push tasks without a working URL
- **Same split, same folds** for every comparison; no hardcoded comparison constants
- **No in-sample metrics**: calibration and thresholds are evaluated on data not used to fit them
- **Report negative results** honestly (failed objectives stay visible)
- Avoid scope creep: anything not in PRD goes to "Future Work" in `02_PRD.md`

## 5. Quick Links
- Model review: `docs/13_MODEL_REVIEW.md`
- Improvement plan: `docs/14_IMPROVEMENT_PLAN.md`
- Tech stack: `docs/00_TECH_STACK.md`
- Publishing plan: `docs/10_PUBLISHING_PLAN.md`
- Problem and objectives: `docs/01_PROBLEM_AND_OBJECTIVES.md`
- PRD: `docs/02_PRD.md`
- Plan and timeline: `docs/03_PROJECT_PLAN.md`
- Technical design: `docs/04_TECHNICAL_DESIGN.md`
- Data spec: `docs/05_DATA_SPEC.md`
- Experiment plan: `docs/06_EXPERIMENT_PLAN.md`
- Tasks backlog: `docs/07_TASKS.md`
- Decisions log: `docs/08_DECISIONS_LOG.md`
- Progress log: `docs/09_PROGRESS_LOG.md`

## 6. Kick-off Prompt (copy when starting a session)
```
Read CLAUDE.md, then follow its Read Order. Continue from the latest entry in
docs/09_PROGRESS_LOG.md and execute the next task in docs/07_TASKS.md.
Act as Senior ML Engineer + PM. Update the tasks and progress logs when done.
```
