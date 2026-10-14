import csv

with open("results/kfold_campaign/dice_ci_summary.csv", "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# Look for point_estimates or batch_mean_estimates for overall strata
for r in rows:
    if r.get("stratum") == "overall" and r.get("fold") == "":
        print(f"Overall {r['run_label']}: pt={float(r['point_estimate'])*100:.2f}%, CI=[{float(r['ci_low'])*100:.2f}, {float(r['ci_high'])*100:.2f}]")

# Also look at rows where fold != ""
print("\nSample fold rows:")
for r in rows:
    if r.get("fold") != "":
        print(f"Fold {r['run_label']} fold={r['fold']} stratum={r['stratum']}: pt={float(r['point_estimate'])*100:.2f}%")
        break
