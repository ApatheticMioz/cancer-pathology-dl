import csv

with open("results/kfold_campaign/dice_ci_summary.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    dice_rows = list(reader)

print(f"Total rows in dice_ci_summary.csv: {len(dice_rows)}")
print(f"Columns: {reader.fieldnames}")

# Let's inspect the across-folds / overall rows
by_run = {}
for r in dice_rows:
    run_name = r.get("run_name") or r.get("run") or r.get("experiment")
    scope = r.get("scope") or r.get("fold") or r.get("level")
    by_run.setdefault(run_name, []).append(r)

print(f"Total distinct runs in dice_ci_summary: {len(by_run)}")
for run_name, entries in list(by_run.items())[:10]:
    print(f"Run: {run_name} (entries: {len(entries)})")
    for e in entries:
        print("  ", {k: v for k, v in e.items() if k in ["run_name", "fold", "scope", "mean", "dice_mean", "ci_lower", "ci_upper", "metric", "estimator"]})
