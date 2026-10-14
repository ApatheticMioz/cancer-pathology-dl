import re

with open("paper/all_dice_no_slice.tex", "r", encoding="utf-8") as f:
    text = f.read()

# Find all occurrences of \ref{...}
for m in re.finditer(r'([^\n.?!;]{0,50}\\ref\{([^}]+)\}[^\n.?!;]{0,50})', text):
    snippet, target = m.group(1), m.group(2)
    print(f"Target: {target:25s} | Snippet: {snippet.strip()}")
