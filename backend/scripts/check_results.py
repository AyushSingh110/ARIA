import json
from pathlib import Path

results = []
for f in sorted(Path("data/gaia/results").glob("*.json")):
    r = json.loads(f.read_text(encoding="utf-8"))
    if not r.get("run_error"):
        results.append(r)

print("=== req_sat=0 cases (pipeline fix check) ===")
for r in results:
    if r.get("requirement_satisfaction", 1.0) == 0.0:
        tid = r["gaia_task_id"][:8]
        label = r["aria_label"]
        correct = r["gaia_correct"]
        print(f"  {tid}  label={label}  correct={correct}")

print()
print("=== Correct answers (gaia_correct=True) ===")
for r in results:
    if r.get("gaia_correct"):
        label = r["aria_label"] or "none"
        expected = r["expected_answer"]
        out = r["executor_output"][:70]
        print(f"  [{label:22}] expected={expected}  out={out}")

print()
print("=== False-clean: ARIA said none but agent was WRONG ===")
fc = [r for r in results if not r.get("aria_label") and r.get("gaia_correct") == False]
for r in fc:
    sat = r.get("requirement_satisfaction", 0)
    expected = r["expected_answer"]
    tid = r["gaia_task_id"][:8]
    q = r.get("question", "")[:60]
    print(f"  {tid}  req_sat={sat:.2f}  expected={expected}  q={q}")

print()
print(f"Summary: {len(results)} results | false-clean: {len(fc)} | true-clean: {sum(1 for r in results if not r.get('aria_label') and r.get('gaia_correct'))}")
