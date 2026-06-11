# ARIA — Execution Status

> Last updated: 2026-06-11 · Internal document — do not publish.

---

## 1. What has been executed (this build-out)

### Pipeline fixes (research-driven)
| Item | File | Status |
|---|---|---|
| tool_misuse over-prediction fix — requires actual error evidence (`tool_error_loop` flag or "Error" in trace) before assigning tool_misuse | `backend/aria/agents/diagnostician.py`, `backend/api/main.py` | ✅ done + verified |
| req_sat=0 labeled "none" bug — deterministic overrides after DSPy call | `backend/aria/agents/diagnostician.py` | ✅ done + verified on GAIA rerun |
| Legacy 4-field compiled DSPy program no longer loaded (was overriding disambiguation) — only v2 program or zero-shot | `backend/aria/agents/diagnostician.py` | ✅ done |
| Agreement metric bug — `None` vs `"none"` mismatch undercounted agreement (reported 38%, actual **68%**) | `backend/scripts/analyze_realbench.py` | ✅ fixed |

### Critic v3 — factual grounding (new research contribution)
| Item | File | Status |
|---|---|---|
| Grounding module: extract central claim → independent DuckDuckGo search → verify supported/contradicted/unverifiable | `backend/aria/agents/grounding.py` | ✅ done |
| Wired into Diagnostician: clean-looking runs (req_sat ≥ 0.75, no flags) with contradicted claims → `hallucination_loop` | `backend/aria/agents/diagnostician.py` | ✅ done |
| Config flag `GROUNDING_ENABLED` (on in `.env`) | `backend/aria/config/settings.py` | ✅ done |
| `grounding` field added to `ARIAState` | `backend/aria/state/schema.py` | ✅ done |

### DSPy Diagnostician v2 recompile
| Item | File | Status |
|---|---|---|
| Recompile script using human-labeled real data (48 usable examples: 38 train / 10 val), BootstrapFewShot, saves `diagnostician_v2.json`, runs held-out validation | `backend/scripts/recompile_diagnostician_v2.py` | ✅ script done + dry-run verified — **you must run the compile (Groq API calls)** |

### Frontend dashboard (React + Vite + Recharts)
| Item | Status |
|---|---|
| Stat cards: total runs, avg req satisfaction, pass rate, most common failure, human agreement | ✅ |
| Failure-distribution donut chart (light research palette) | ✅ |
| Recent failures table | ✅ |
| Diagnose panel: paste task + tool calls + output → live diagnosis with requirement checklist, evidence, suggested action | ✅ |
| Feedback buttons (correct / wrong + correction) wired to `/feedback` | ✅ |
| API proxy `/api → localhost:8000`, auto-refresh every 15 s, production build verified | ✅ |

### Research figures notebook
| Item | Status |
|---|---|
| `research/figures.ipynb` — executed end-to-end, 6 figures at 300 DPI in `research/figures/` | ✅ |
| Fig 1: ARIA vs Human distribution (RealBench) · Fig 2: GAIA distribution + diagnostic correlation · Fig 3: Critic evolution 8→42→68% · Fig 4: req-sat histograms · Fig 5: confusion matrix · Fig 6: hallucination blind spot | ✅ |
| `research/figures/public_numbers.json` — the only numbers safe to publish | ✅ |

### Packaging / SDK
| Item | Status |
|---|---|
| `backend/pyproject.toml` — installable as `aria-agent-diagnostics`, `aria` CLI entry point, `[api]` and `[dev]` extras | ✅ |
| `aria/sdk.py` — `diagnose()` (in-process), `diagnose_remote()` (HTTP), `run_task()` (full pipeline) | ✅ imports verified |
| `ddgs` + fastapi/uvicorn added to env; `ddgs` added to requirements.txt | ✅ |

### IP protection (repo can go public)
| Item | Status |
|---|---|
| `.gitignore`: `backend/data/realbench/`, `backend/data/gaia/`, `backend/data/api_runs/`, `backend/data/synthetic/`, `research/`, `node_modules/`, experience store | ✅ |
| 126 already-tracked private files untracked via `git rm --cached` (files stay on disk) | ✅ |
| `.env` (API keys) confirmed never tracked; `.env.example` created | ✅ |
| README shows aggregate numbers only — no raw JSON, no labeled traces, no full analyses | ✅ |

> ⚠️ **CRITICAL before making the repo public:** the private data still exists in **git history** (commits `43129bb`, `5aebeb7`, `ca2e64d`, …). Untracking only affects future commits. Two options:
> 1. **Fresh public repo (recommended, simple):** create a new repo, copy the working tree, single initial commit. History stays private.
> 2. `git filter-repo` to rewrite history — error-prone, do only if you must keep history.
> Also: your **GROQ API key and HF token are in `.env`** — rotate them if there is any chance they were ever committed or shared.

---

## 2. Updated headline numbers (after fixes)

| Metric | Value |
|---|---|
| Human–ARIA agreement (RealBench, 50 labeled) | **68%** (was misreported as 38% due to metric bug) |
| GAIA failure-detection precision | **91.7%** |
| GAIA false-clean cases | 4/41, of which 3 had req_sat = 1.0 (hallucination blind spot → Critic v3) |
| Critic evolution | 8% (v1) → 42% (v2) → 68% (v2 + disambiguation) |
| Labeled training examples ready for DSPy v2 | 48 (38 train / 10 val) |

---

## 3. Commands — run these in order

All backend commands from `C:\Users\ASUS\Desktop\ARIA\backend` with the aria env:
`C:\Users\ASUS\anaconda3\envs\aria\python.exe` (alias as `$py` below).

### A. Recompile Diagnostician v2 on your labeled data (DO THIS FIRST — uses Groq API)
```powershell
cd C:\Users\ASUS\Desktop\ARIA\backend
C:\Users\ASUS\anaconda3\envs\aria\python.exe scripts/recompile_diagnostician_v2.py --dry-run   # preview
C:\Users\ASUS\anaconda3\envs\aria\python.exe scripts/recompile_diagnostician_v2.py             # compile + held-out validation
```
The pipeline auto-loads `data/compiled/diagnostician_v2.json` afterwards.

### B. Re-validate after recompile (research before/after numbers)
```powershell
# Rerun GAIA with v2 program + Critic v3 grounding active
C:\Users\ASUS\anaconda3\envs\aria\python.exe scripts/gaia_run_batch.py --all --force --delay 10 --batch-delay 30
C:\Users\ASUS\anaconda3\envs\aria\python.exe scripts/gaia_agreement.py --save

# Recompute RealBench agreement (fixed metric)
C:\Users\ASUS\anaconda3\envs\aria\python.exe scripts/analyze_realbench.py
```

### C. Manual review / labeling (when new runs accumulate)
```powershell
C:\Users\ASUS\anaconda3\envs\aria\python.exe scripts/review_realbench.py     # label RealBench runs
C:\Users\ASUS\anaconda3\envs\aria\python.exe scripts/gaia_run_batch.py --status
```

### D. Regenerate research figures (after any data change)
```powershell
cd C:\Users\ASUS\Desktop\ARIA\research
C:\Users\ASUS\anaconda3\envs\aria\python.exe -m jupyter nbconvert --to notebook --execute --inplace figures.ipynb
# figures land in research/figures/*.png (300 DPI) + public_numbers.json
# or open interactively:
C:\Users\ASUS\anaconda3\envs\aria\python.exe -m jupyter notebook figures.ipynb
```

### E. Run the full product (API + dashboard)
```powershell
# Terminal 1 — API
cd C:\Users\ASUS\Desktop\ARIA\backend
C:\Users\ASUS\anaconda3\envs\aria\python.exe -m uvicorn api.main:app --port 8000

# Terminal 2 — dashboard
cd C:\Users\ASUS\Desktop\ARIA\frontend
npm run dev
# open http://localhost:5173
```

### F. Install as a package (SDK)
```powershell
cd C:\Users\ASUS\Desktop\ARIA\backend
C:\Users\ASUS\anaconda3\envs\aria\python.exe -m pip install -e ".[api]"
```

---

## 4. What is still left

| Item | Effort | Notes |
|---|---|---|
| **Run the DSPy v2 recompile** (command A) | 10–20 min | Script ready; needs your Groq quota |
| **Rerun GAIA with v2 + grounding** (command B) | ~1–2 h | Produces the paper's before/after table |
| Validate Critic v3 on the 6 known hallucination cases | 30 min | Check the 3 GAIA + 3 RealBench false-cleans now get `hallucination_loop` |
| GAIA Level 2 batch (10–15 tasks) | ~2 h | `gaia_download.py --level 2` then batch runner |
| Manual labeling of the 41 GAIA runs | 2–3 h | Adds ~40 training examples for the next recompile |
| Taxonomy v2 (mechanism/outcome layers) | research | Design doc + schema change |
| Ablation study (ARIA vs no-ARIA decision support) | research | Paper section |
| Paper draft | writing | Figures + numbers are ready |
| Fresh public repo (history clean) | 30 min | See warning in section 1 |
| PyPI publish of `aria-agent-diagnostics` | 1 h | After repo is public; reserve the name early |

---

## 5. Database recommendation (MongoDB question)

**Not yet — stay file-based for now.** Reasons:

- Current volume (~150 JSON files) is trivially handled by the filesystem; MongoDB adds setup, hosting, and auth complexity with zero research benefit at this scale.
- Your dashboard, analysis scripts, and labeling tools all read JSON directly — migrating now means rewriting them all mid-research-cycle.

**Add a DB when one of these happens:**
1. ARIA runs as a hosted service where multiple users submit traces concurrently (file writes will race),
2. you exceed ~5,000 stored runs, or
3. you need queries like "all hallucination cases with req_sat > 0.8 across benchmarks" frequently.

When that day comes: **MongoDB Atlas free tier** fits the JSON-document shape of your records perfectly, and the migration is one script (insert every JSON file as a document). The `experience_store.json` would become a collection too. SQLite is the lighter alternative if you stay single-user.

---

## 6. Files created/changed in this build-out

**New:** `aria/agents/grounding.py` · `aria/sdk.py` · `scripts/recompile_diagnostician_v2.py` · `backend/pyproject.toml` · `backend/.env.example` · `research/figures.ipynb` (+6 PNGs + public_numbers.json) · `frontend/` (full React app: package.json, vite.config.js, index.html, src/App.jsx, src/api.js, src/constants.js, src/styles.css, 4 components) · `EXECUTION_STATUS.md`

**Modified:** `aria/agents/diagnostician.py` (tool_misuse fix, grounding hook, v2-only program loading) · `api/main.py` (tool_misuse fix in post-processing) · `aria/config/settings.py` (GROUNDING_ENABLED) · `aria/state/schema.py` (grounding field) · `scripts/analyze_realbench.py` (agreement metric fix) · `scripts/gaia_run_batch.py` (abbreviation matching) · `requirements.txt` (ddgs) · `.gitignore` (private data) · `README.md` (full rewrite) · `.env` (GROUNDING_ENABLED=true)
