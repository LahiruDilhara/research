import re
import os
import subprocess

# 1. Parse all bib keys in research-db/references.bib
with open('research-db/references.bib', 'r') as f:
    bib_text = f.read()

bib_keys = set(re.findall(r'@\w+\s*\{\s*([\w\-]+)\s*,', bib_text))
print(f"Total BibTeX keys in references.bib: {len(bib_keys)}")

# 2. Parse all \cite{...} in all chapter tex files
chapter_files = [f"chapters/{f}" for f in sorted(os.listdir('chapters')) if f.endswith('.tex')]
missing_cites = set()
used_cites = set()

for cfile in chapter_files:
    with open(cfile, 'r') as f:
        content = f.read()
    cites = re.findall(r'\\cite\{([^}]+)\}', content)
    for cgroup in cites:
        keys = [k.strip() for k in cgroup.split(',')]
        for k in keys:
            used_cites.add(k)
            if k not in bib_keys:
                missing_cites.add((cfile, k))

print(f"Total unique citation keys used across chapters: {len(used_cites)}")
print(f"Missing citation keys ({len(missing_cites)}):")
for cf, k in missing_cites:
    print(f"  In {cf}: {k}")

# 3. Parse all \ref{...} and \label{...} across chapter files
labels = set()
refs = set()
for cfile in chapter_files:
    with open(cfile, 'r') as f:
        content = f.read()
    found_labels = re.findall(r'\\label\{([^}]+)\}', content)
    found_refs = re.findall(r'\\ref\{([^}]+)\}', content)
    labels.update(found_labels)
    for r in found_refs:
        refs.add((cfile, r))

missing_refs = [(cf, r) for cf, r in refs if r not in labels]
print(f"\nMissing label references ({len(missing_refs)}):")
for cf, r in missing_refs:
    print(f"  In {cf}: {r}")

