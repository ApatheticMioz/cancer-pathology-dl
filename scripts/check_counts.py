import json
import glob
from pathlib import Path

kfold_files = sorted(glob.glob("results/kfold_campaign/kfold_*.json"))
print(f"Total kfold summary JSONs found: {len(kfold_files)}")

dataset_counts = {}
for kf in kfold_files:
    with open(kf, "r") as f:
        d = json.load(f)
    for run_name, run_data in d.get("runs", {}).items():
        ds = run_data["dataset"]
        enc = run_data["encoder"]
        fold_results = run_data.get("fold_results", [])
        if fold_results:
            n_samples = fold_results[0].get("samples")
            n_train = fold_results[0].get("train_samples")
            n_val = fold_results[0].get("val_samples")
            dataset_counts[ds] = (n_samples, n_train, n_val)
        mean_acc = run_data.get("mean_val_acc")
        mean_dice = run_data.get("mean_val_dice")
        # print first fold and mean
        # print(f"{Path(kf).stem}: {ds} {enc} samples={n_samples} acc={mean_acc*100:.2f}% dice={mean_dice*100:.2f}%")

for ds, counts in dataset_counts.items():
    print(f"Dataset {ds}: total_samples={counts[0]}, train_samples={counts[1]}, val_samples={counts[2]}")
