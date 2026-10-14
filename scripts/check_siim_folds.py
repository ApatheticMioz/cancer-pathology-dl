import json
import glob

# Check SIIM kfold JSONs
siim_kfold_files = sorted(glob.glob("results/kfold_campaign/kfold_*siim*.json"))
for kf in siim_kfold_files:
    with open(kf, "r") as f:
        d = json.load(f)
    for run_name, run_data in d.get("runs", {}).items():
        print(f"\nFile: {kf}")
        print(f"Run: {run_name}, mean_val_acc={run_data.get('mean_val_acc')*100:.2f}%, mean_val_dice={run_data.get('mean_val_dice')*100:.2f}%")
        for fr in run_data.get("fold_results", []):
            print(f"  Fold {fr['fold_idx']}: val_acc={fr.get('final_val_acc')*100:.2f}%, val_dice={fr.get('final_val_dice')*100:.2f}%")
