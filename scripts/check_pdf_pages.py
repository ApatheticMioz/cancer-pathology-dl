import pypdf

reader = pypdf.PdfReader('paper/all_dice_no_slice.pdf')
print('Num pages:', len(reader.pages))
for i in range(len(reader.pages)):
    lines = reader.pages[i].extract_text().splitlines()
    print(f'Page {i+1}: {len(lines)} lines | First: {lines[0] if lines else ""} | Last: {lines[-1] if lines else ""}')
