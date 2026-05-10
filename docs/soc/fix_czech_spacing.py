#!/usr/bin/env python3
import re
from pathlib import Path

PREPS = frozenset('ksvzouaiKSVZOUAI')

def fix(path, dry=True):
    text = Path(path).read_text(encoding='utf-8')
    lines = text.split('\n')
    fixed = 0
    out = []
    
    for line in lines:
        new = line
        for p in PREPS:
            rx = rf'(?<![\\{{])\b{p}\b\s+(?=[a-zA-Z])'
            new = re.sub(rx, lambda m: m.group() + '~' if m.group() else m.group(), new, flags=re.IGNORECASE)
        if new != line:
            fixed += 1
        out.append(new)
    
    if not dry:
        Path(path).write_text('\n'.join(out), encoding='utf-8')
    
    print(f"{'DRY ' if dry else ''}Fixed {fixed}/{len(lines)} lines")

if __name__ == '__main__':
    import sys
    dry = '--dry' in sys.argv or '-n' in sys.argv
    path = [a for a in sys.argv[1:] if not a.startswith('--')][0] if len(sys.argv) > 1 else 'soc.tex'
    fix(path, dry)