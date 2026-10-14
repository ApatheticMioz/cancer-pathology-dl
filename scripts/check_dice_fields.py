import csv

with open("results/kfold_campaign/dice_ci_summary.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    print("Fieldnames:", reader.fieldnames)
    for i, row in enumerate(reader):
        if row.get("fold") == "" or row.get("fold") == "overall" or row.get("fold") is None or i < 5:
            print(row)
