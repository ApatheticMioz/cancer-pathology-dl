import re

with open('paper/references.bib', 'r', encoding='utf-8') as f:
    bib_str = f.read()

entries = re.findall(r'@(\w+)\s*\{\s*([^,]+),\s*(.*?)\n\}', bib_str, re.DOTALL)
print(f"Total entries: {len(entries)}")
for i, (entry_type, key, fields_str) in enumerate(entries):
    fields = {}
    for line in fields_str.split('\n'):
        line = line.strip()
        m = re.match(r'(\w+)\s*=\s*[\"{](.*)[\"}],?', line)
        if m:
            fields[m.group(1).lower()] = m.group(2).rstrip('"},')
    print(f"#{i+1} [{key}] ({entry_type})")
    print(f"  Title: {fields.get('title')}")
    print(f"  Author: {fields.get('author')}")
    print(f"  Venue: {fields.get('journal') or fields.get('booktitle') or fields.get('howpublished')}")
    print(f"  Year: {fields.get('year')}, Vol: {fields.get('volume')}, Number: {fields.get('number')}, Pages: {fields.get('pages')}")
    print(f"  DOI/URL: {fields.get('doi') or fields.get('url') or fields.get('howpublished')}")
