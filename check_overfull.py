import re

with open('out/main.log', 'r', errors='ignore') as f:
    log_text = f.read()

overfulls = re.findall(r'Overfull \\hbox \(([\d\.]+)pt too wide\) in paragraph at lines (\d+)--(\d+)(.*?)(?=\n\n|\n[A-Z]|\Z)', log_text, re.DOTALL)
print(f"Total Overfull \hbox warnings: {len(overfulls)}")
for pts, l1, l2, snippet in overfulls:
    clean_snippet = snippet.replace('\n', ' ')[:120]
    print(f"Lines {l1}-{l2} ({pts}pt too wide): {clean_snippet}")
