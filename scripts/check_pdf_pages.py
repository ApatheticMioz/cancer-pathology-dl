import subprocess

def get_page_count():
    res = subprocess.run(["pdfinfo", "paper/all_dice_no_slice.pdf"], capture_output=True, text=True)
    for line in res.stdout.splitlines():
        if "Pages:" in line:
            return int(line.split()[1])
    return 0

num_pages = get_page_count()
print("Num pages:", num_pages)
for i in range(1, num_pages + 1):
    res = subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), "paper/all_dice_no_slice.pdf", "-"], capture_output=True, text=True)
    lines = [l for l in res.stdout.splitlines() if l.strip()]
    first_line = lines[0] if lines else ""
    last_line = lines[-1] if lines else ""
    print(f"Page {i}: {len(lines)} non-empty lines | First: {first_line[:40]} | Last: {last_line[:40]}")
