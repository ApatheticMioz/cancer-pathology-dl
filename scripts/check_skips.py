import json
import glob

kfold_files = sorted(glob.glob("results/kfold_campaign/kfold_*.json"))
total_skips = 0
skips_by_run = {}

for kf in kfold_files:
    with open(kf, "r") as f:
        d = json.load(f)
    for run_name, run_data in d.get("runs", {}).items():
        fold_results = run_data.get("fold_results", [])
        run_skips = 0
        for fr in fold_results:
            skips = fr.get("nonfinite_grad_skips", 0)
            run_skips += skips
        skips_by_run[kf] = run_skips
        total_skips += run_skips

print(f"Total non-finite gradient skips across all kfold files: {total_skips}")
for kf, skips in skips_by_run.items():
    if skips > 0:
        print(f"  {kf}: {skips} skips")
