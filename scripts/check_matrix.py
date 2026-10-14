import csv
import json
import glob
from pathlib import Path

# Load paper_results_matrix_with_ci.csv
with open("paper/paper_results_matrix_with_ci.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total rows in paper_results_matrix_with_ci.csv: {len(rows)}")
for r in rows:
    label = r["Run Label"]
    ds = r["Dataset"]
    acc_m = r["Kfold Acc Mean (%)"]
    dice_m = r["Kfold Dice Mean (%)"]
    dice_ci_l = r["Dice 95% CI Lower (bootstrap)"]
    dice_ci_u = r["Dice 95% CI Upper (bootstrap)"]
    print(f"{label:35s} | {ds:8s} | Acc: {acc_m:5s}% | Dice: {dice_m:5s}% | Dice CI: [{dice_ci_l}, {dice_ci_u}]")
