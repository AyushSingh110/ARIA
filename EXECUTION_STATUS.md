# ARIA — Execution Status

> Last updated: 2026-06-12 · Internal document — do not publish.

---

## 1. Current verified numbers (after your reruns)

| Metric | Value | Change |
|---|---|---|
| Human–ARIA agreement (RealBench, 50 labeled) | **78%** (39/50) | ↑ from 68% — tool_misuse fix + v2 recompile worked |
| GAIA failure-flag precision | **91.2%** | stable |
| GAIA tool_misuse over-prediction | **eliminated** (28% → 0%) | fix verified on rerun |
| Diagnostician v2 held-out accuracy | 50% (5/10) | small val set — see §4 |
| GAIA runs OK / errored | 36 / 6 | 6 failed on Groq 429 — retry below |
| Critic evolution | 8% → 42% → **78%** | updated |

Remaining RealBench disagreements (11) are mostly ARIA over-flagging (`ARIA=failure, human=none` ×6) and the 2 taxonomy-gap cases — see `research/RESEARCH_ROADMAP.md` for how these feed taxonomy v2.

---

## 2. What was executed in this build-out (2026-06-12)

### Package → PyPI-ready as **`ariadx`**
| Item | Status |
|---|---|
| Name `ariadx` confirmed available on PyPI ("ARIA + dx (diagnosis)") | ✅ |
| `pyproject.toml` — full metadata: Apache-2.0, classifiers, URLs, keywords, email | ✅ |
| `backend/PYPI_README.md` — PyPI landing page (product-focused, short) | ✅ |
| CLI entry-point bug fixed — `main:run` wasn't packaged; CLI moved to `aria/cli.py`, entry = `aria.cli:run` | ✅ verified |
| Wheel built + contents audited — **only source modules, zero data files** | ✅ `dist/ariadx-0.1.0-py3-none-any.whl` |
| `backend/PUBLISHING.md` — complete step-by-step publish guide (TestPyPI rehearsal → real) | ✅ |

### Open-source documentation (complete set)
| File | Status |
|---|---|
| `LICENSE` — official Apache 2.0 text | ✅ |
| `CONTRIBUTING.md` — setup, PR rules, adapter guide, taxonomy context | ✅ |
| `CODE_OF_CONDUCT.md` — Contributor Covenant 2.1 | ✅ |
| `SECURITY.md` — reporting, scope (file sandbox, no-auth API warning) | ✅ |
| `docs/README.md` — docs index | ✅ |
| `docs/getting-started.md` — install → first diagnosis in 5 min | ✅ |
| `docs/sdk-reference.md` — all 3 SDK functions + CLI, full schemas | ✅ |
| `docs/api-reference.md` — all 6 endpoints with request/response JSON | ✅ |
| `docs/failure-taxonomy.md` — 5 classes, detection, gaps, v2 motivation | ✅ |
| `docs/architecture.md` — pipeline deep-dive + stack table | ✅ |
| `docs/adapters.md` — LangGraph/OpenAI usage + write-your-own guide | ✅ |
| `README.md` — full professional rewrite: badges, new 78% numbers, doc links | ✅ |

### Pipeline weak points fixed
| Item | Status |
|---|---|
| `grounding` field was **never persisted** in GAIA results (Critic v3 unverifiable) — now saved by `gaia_run_batch.py` | ✅ |
| `--retry-failed` flag added to `gaia_run_batch.py` — reruns only 429-errored tasks | ✅ |
| `--from-batch/--to-batch` range support (from previous session) | ✅ |
| Resumable `--validate-all` with checkpoint in recompile script (from previous session) | ✅ |

### Research planning
| Item | Status |
|---|---|
| `research/RESEARCH_ROADMAP.md` — detailed prioritized research plan (private) | ✅ |

---

## 3. Commands — what to run next, in order

`$py = C:\Users\ASUS\anaconda3\envs\aria\python.exe`, run from `backend/`.

### A. Retry the 6 rate-limited GAIA runs (~20 min)
```powershell
cd C:\Users\ASUS\Desktop\ARIA\backend
C:\Users\ASUS\anaconda3\envs\aria\python.exe scripts/gaia_run_batch.py --retry-failed --delay 15
C:\Users\ASUS\anaconda3\envs\aria\python.exe scripts/gaia_agreement.py --save
```

### B. Publish to PyPI (follow `backend/PUBLISHING.md` for full detail)
```powershell
cd C:\Users\ASUS\Desktop\ARIA\backend
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
C:\Users\ASUS\anaconda3\envs\aria\python.exe -m build
C:\Users\ASUS\anaconda3\envs\aria\python.exe -m twine check dist/*
# rehearsal: twine upload --repository testpypi dist/*
C:\Users\ASUS\anaconda3\envs\aria\python.exe -m twine upload dist/*
```

### C. Label the 36 GAIA runs (research priority 1.1, ~2–3 h)
Edit `human_label` + `reviewed: true` in `data/gaia/results/*.json`, or adapt `scripts/review_realbench.py`.

### D. Recompile Diagnostician on combined data (after C)
```powershell
C:\Users\ASUS\anaconda3\envs\aria\python.exe scripts/recompile_diagnostician_v2.py --dry-run
C:\Users\ASUS\anaconda3\envs\aria\python.exe scripts/recompile_diagnostician_v2.py
C:\Users\ASUS\anaconda3\envs\aria\python.exe scripts/recompile_diagnostician_v2.py --validate-all   # resumable
```

### E. Regenerate figures with new numbers (78%)
```powershell
cd C:\Users\ASUS\Desktop\ARIA\research
C:\Users\ASUS\anaconda3\envs\aria\python.exe -m jupyter nbconvert --to notebook --execute --inplace figures.ipynb
```

### F. Run the product
```powershell
# API:        cd backend;  python -m uvicorn api.main:app --port 8000
# Dashboard:  cd frontend; npm run dev
```

---

## 4. Known weak points still open

| Item | Why it matters | Plan |
|---|---|---|
| Held-out accuracy 50% on 10 examples | Too noisy to publish; small val set | Recompile after GAIA labeling (~85 examples, stratified split) — roadmap 1.2 |
| Critic v3 grounding never validated end-to-end | Core paper claim unverified | Roadmap 1.3 — re-diagnose the 6 known hallucination cases with grounding on |
| 6 GAIA runs errored (429) | N=36 instead of 42 | Command A above |
| ARIA over-flags clean runs (6/11 disagreements) | Precision on "none" class | Analyze after GAIA labels; candidate disambiguation rule |
| `figures.ipynb` shows old numbers (68%) | Stale figures | Command E |
| Git history contains private data | **Blocker for public repo** | Fresh repo with single initial commit (see §5) |
| GROQ key + HF token in `.env` | Leak risk | **Rotate both before going public** |

---

## 5. Before making the repo public — checklist

1. **Rotate** GROQ_API_KEY and HUGGING_FACE_TOKEN (console.groq.com / hf.co settings)
2. **Fresh repo**: create new GitHub repo → copy working tree (without `.git`, `backend/data/`, `research/`, `.env`) → single initial commit. Old history stays private.
3. Verify `.gitignore` excludes: `backend/data/realbench|gaia|api_runs|synthetic`, `research/`, `.env`, `node_modules/`, `dist/`
4. `pip install ariadx` works (after B)
5. Add repo topics on GitHub: `llm-agents`, `agent-evaluation`, `ai-reliability`, `failure-detection`

---

## 6. Database (MongoDB) — unchanged recommendation

Stay file-based. Add MongoDB Atlas (free tier) only when: concurrent users, >5,000 runs, or frequent cross-benchmark queries. Migration is one script.
