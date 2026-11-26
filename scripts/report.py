#!/usr/bin/env python3
"""Collect actual runs only; no estimated results or mixing boundaries."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs');p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rows=[]
for f in sorted(Path(a.runs).rglob('result.json')):
    r=json.loads(f.read_text());c=r.get('config',{});lat=r.get('latency_ms',{})
    if 'throughput_fps' not in r: continue
    rows.append({'run':str(f.parent),'machine':r.get('machine'),'model_sha256':r.get('model_sha256'),
        'input_sha256':r.get('input_sha256'),'mode':c.get('mode'),'backend':c.get('backend','cpu'),
        'threads':c.get('threads'),'profile':r.get('profile_instrumented'), 'status':r.get('status'),
        'images':r.get('images'),'elapsed_s':r.get('elapsed_s'),'throughput_fps':r.get('throughput_fps'),
        'p50_ms':lat.get('p50'),'p95_ms':lat.get('p95'),'p99_ms':lat.get('p99'),
        'session_init_s':r.get('session_init_s'),'first_call_s':r.get('first_call_s')})
(out/'report.json').write_text(json.dumps({'schema_version':1,'runs':rows,'warning':'Compare only same data, shape, precision, timing boundary and accuracy-passing runs. Profiled runs excluded from final KPI.'},indent=2)+'\n')
if rows:
    with (out/'report.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(f'{len(rows)} actual runs')
