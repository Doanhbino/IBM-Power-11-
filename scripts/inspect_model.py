#!/usr/bin/env python3
"""Inspect names, types, static/dynamic shapes and external tensor hashes before configuring."""
import argparse,json
from collections import Counter
from pathlib import Path
import onnx
from common import model_files
p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--check',action='store_true');p.add_argument('--output');a=p.parse_args()
m=onnx.load(a.model,load_external_data=False)
def tensor(v):
    t=v.type.tensor_type
    return {'name':v.name,'dtype':onnx.TensorProto.DataType.Name(t.elem_type),'shape':[d.dim_value if d.HasField('dim_value') else d.dim_param or None for d in t.shape.dim]}
if a.check:onnx.checker.check_model(a.model,full_check=True)
r={'opsets':{x.domain:x.version for x in m.opset_import},'inputs':[tensor(x) for x in m.graph.input],'outputs':[tensor(x) for x in m.graph.output],
   'operators':dict(Counter(x.op_type for x in m.graph.node)),'metadata':{x.key:x.value for x in m.metadata_props},'files':model_files(a.model),'checker':bool(a.check)}
s=json.dumps(r,ensure_ascii=False,indent=2)
if a.output:Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(s+'\n')
print(s)
