#!/usr/bin/env python3
"""Hash immutable dataset files for identical split/image verification."""
import argparse,json
from pathlib import Path
from common import sha256
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.root).resolve();out=Path(a.output).resolve()
rows=[]
for f in sorted(root.rglob('*')):
    if f.is_symlink(): raise RuntimeError('Dataset must not contain symlinks')
    if f.is_file() and f.resolve()!=out:
        rows.append({'path':str(f.relative_to(root)),'bytes':f.stat().st_size,'sha256':sha256(f)})
if not rows: raise RuntimeError('Empty dataset')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'root_label':root.name,'files':rows},indent=2)+'\n')
print(f'{len(rows)} files; manifest SHA256 {sha256(out)}')
