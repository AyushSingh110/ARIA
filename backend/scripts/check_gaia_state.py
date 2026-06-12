"""Quick state check of GAIA results after rerun."""
import json
from pathlib import Path

results = sorted(Path("data/gaia/results").glob("*.json"))
print(f"result files: {len(results)}")

errs, ok, none_label = [], 0, 0
halluc, grounded = [], 0
labels = {}
for f in results:
    r = json.loads(f.read_text(encoding="utf-8"))
    if r.get("run_error"):
        errs.append((f.stem[:14], str(r["run_error"])[:90]))
        continue
    ok += 1
    lab = r.get("aria_label") or "none"
    labels[lab] = labels.get(lab, 0) + 1
    if lab == "hallucination_loop":
        halluc.append(f.stem[:14])
    if r.get("grounding"):
        grounded += 1

print(f"ok: {ok}, errors: {len(errs)}")
print("label distribution:", labels)
print(f"runs with grounding data: {grounded}, hallucination_loop: {halluc}")
print("\nERRORS:")
for name, msg in errs:
    print(f"  {name}: {msg}")
