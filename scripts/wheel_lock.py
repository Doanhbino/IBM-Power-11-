#!/usr/bin/env python3
"""Create hash-pinned requirements from an already resolved, reviewed wheelhouse."""
import argparse,email,json,re
from pathlib import Path
from zipfile import ZipFile
import hashlib

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
p=argparse.ArgumentParser();p.add_argument('wheelhouse');p.add_argument('--output',required=True);a=p.parse_args()
rows={}
for f in sorted(Path(a.wheelhouse).glob('*.whl')):
    with ZipFile(f) as z:
        names=[n for n in z.namelist() if n.endswith('.dist-info/METADATA')]
        if len(names)!=1: raise ValueError(f'Unexpected wheel METADATA: {f.name}')
        meta=email.message_from_bytes(z.read(names[0]))
    name=re.sub(r'[-_.]+','-',meta['Name']).lower();version=meta['Version']
    if name in rows: raise ValueError(f'Duplicate distribution {name}; use a clean wheelhouse with one wheel/package')
    rows[name]=f'{name}=={version} --hash=sha256:{sha256(f)}'
if not rows: raise ValueError('No wheels')
Path(a.output).write_text('# Generated from reviewed wheel bytes; runtime/ABI correctness requires target smoke tests.\n'+'\n'.join(rows[k] for k in sorted(rows))+'\n')
print(json.dumps({'packages':len(rows),'lock':a.output,'note':'Does not establish provenance or compatibility. pip check + smoke/accuracy gates still required.'}))
