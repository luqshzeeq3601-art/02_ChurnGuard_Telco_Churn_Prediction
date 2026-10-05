# 09. Progress Log

> Newest entry on top. Each session: Done, Next, Blockers. Weekly PM review every Sunday.

## Current Status
| Item | Value |
|---|---|
| Current phase | Phase 1: Data Understanding + EDA (W1) |
| Next task | **T1.1** `validate.py` with pandera schema |
| Overall progress | 6 / 47 tasks |
| Health | On track |

---

## 05 Oct 2026: Phase 0 Setup Completed
- **Done**:
  - **T0.1**: Initialized git repository, added `.gitignore` (data/, models/, mlruns/, .venv/, __pycache__/, coverage).
  - **T0.2**: Created `pyproject.toml` and `requirements.txt` with pinned versions; installed clean virtual environment.
  - **T0.3**: Created full project structure per `04_TECHNICAL_DESIGN.md` with `__init__.py` files; verified `import churnguard` works.
  - **T0.4**: Created `configs/config.yaml` and `src/churnguard/config.py` loader; verified with unit tests (`test_config.py`).
  - **T0.5**: Created `Makefile` and `.pre-commit-config.yaml` with ruff hooks; `ruff check`, `ruff format`, and `pytest` pass with 82% coverage.
  - **T0.6**: Set up raw data loading in `src/churnguard/data/load.py` for both Telco dataset (7,043 rows) and Malaysia cellular subscribers context dataset (66 rows); verified with `test_load.py`.
- **Next**: Phase 1 — Data Understanding + EDA (starting with T1.1 `validate.py` with pandera schema)
- **Blockers**: None
- **Decisions**: Updated pinned versions in `docs/00_TECH_STACK.md`.

---

## Weekly PM Review Template
```
### Week N Review (date)
- Planned vs done:
- Metrics so far:
- Risks changed:
- Re-plan:
```
