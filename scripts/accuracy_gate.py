#!/usr/bin/env python3
"""Accept explicit absolute COCO AP loss; inputs must be independently evaluated."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--reference-ap',type=float,required=True);p.add_argument('--candidate-ap',type=float,required=True);p.add_argument('--max-drop',type=float,default=.001);p.add_argument('--output',required=True);a=p.parse_args()
if not all(0<=v<=1 for v in (a.reference_ap,a.candidate_ap,a.max_drop)): raise ValueError('AP uses [0,1] units')
r={'reference_ap':a.reference_ap,'candidate_ap':a.candidate_ap,'max_drop':a.max_drop,'drop':a.reference_ap-a.candidate_ap,'passed':a.reference_ap-a.candidate_ap<=a.max_drop}
Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if not r['passed']: raise SystemExit(2)
