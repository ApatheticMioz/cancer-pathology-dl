import json
import glob
import random

def percentile(vals, p):
    # vals must be sorted
    # linear interpolation percentile
    k = (len(vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    d = k - f
    return vals[f] + (vals[c] - vals[f]) * d

def bootstrap_acc_ci(fold_accs, n_resamples=10000, seed=42):
    rng = random.Random(seed)
    n = len(fold_accs)
    boot_means = []
    for _ in range(n_resamples):
        sample = [rng.choice(fold_accs) for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    low = percentile(boot_means, 2.5)
    high = percentile(boot_means, 97.5)
    return low, high

kfold_files = sorted(glob.glob("results/kfold_campaign/kfold_*.json"))

print(f"Total files: {len(kfold_files)}")
for kf in kfold_files:
    with open(kf, "r") as f:
        d = json.load(f)
    for run_name, run_data in d.get("runs", {}).items():
        fold_results = run_data.get("fold_results", [])
        final_accs = [fr.get("final_val_acc") * 100 for fr in fold_results]
        mean_acc = run_data.get("mean_val_acc") * 100
        low, high = bootstrap_acc_ci(final_accs)
        print(f"{kf:60s} | Mean: {mean_acc:6.2f}% | CI: [{low:6.2f}, {high:6.2f}] | Folds: {[round(x, 2) for x in final_accs]}")
