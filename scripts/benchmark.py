#!/usr/bin/env python3
"""Measured host-call latency, model-only or sequential camera/video pipeline."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from common import session, preprocess_bgr, decode, manifest, stats, write_json, sha256
from run_record import identity, runtime_binaries, artifact
from mma_evidence import require_evidence


def run(cfg, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    record = {**identity('inference'), 'status': 'running'}
    write_json(out / 'run.json', record)
    try:
        result = _run(cfg, out, record)
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__)
        write_json(out / 'run.json', record)
        raise
    write_json(out / 'run.json', result)
    return result


def _run(cfg, out, record):
    if int(cfg.get('iterations',100)) < 1 or int(cfg.get('warmup',10)) < 0: raise ValueError('Invalid iterations/warmup')
    cfg=dict(cfg); cfg['profile_prefix']=str(out/'ort')
    result={**manifest(cfg), **record}
    if cfg.get('backend','cpu') == 'cpu':
        import onnxruntime
        result['runtime_binaries'] = runtime_binaries(onnxruntime)
        result['mma_evidence'] = require_evidence(cfg, 'inference', result['model_sha256'],
                                                 result['runtime_binaries'], result['model_files'])
    t=time.perf_counter(); s=session(cfg, bool(cfg.get('profile',False)))
    result['session_init_s']=time.perf_counter()-t
    result['providers']=s.get_providers()
    optimized = out / 'optimized.onnx'
    if optimized.is_file():
        result['optimized_model'] = artifact(optimized)
    result['ort_build_info']=__import__('onnxruntime').get_build_info() if cfg.get('backend')!='tensorrt_native' else None
    if cfg.get('backend')=='tensorrt_native': result['tensorrt_version']=__import__('tensorrt').__version__
    inp=s.get_inputs()[0]; names=[x.name for x in s.get_outputs()]
    mode=cfg.get('mode','model'); cap=None
    if mode=='model':
        with np.load(cfg['input_npz'],allow_pickle=False) as z:
            if set(z.files)!={inp.name}: raise ValueError(f'NPZ must have sole key {inp.name}')
            x=np.ascontiguousarray(z[inp.name])
        if x.ndim!=4: raise ValueError('Input must be NCHW')
        result['input_sha256']=sha256(cfg['input_npz']); result['input_shape']=list(x.shape)
        result['input_dtype']=str(x.dtype)
        if cfg.get('synthetic_input',False): result['result_kind']='synthetic smoke; not application performance'
        else: result['result_kind']='replayed preprocessed tensor'
    elif mode=='camera':
        import cv2
        cv2.setNumThreads(int(cfg.get('opencv_threads',1)))
        source=cfg['source']
        # FFmpeg timeout is set at open time; unsupported backends fail explicitly.
        if isinstance(source,str) and '://' in source:
            cap=cv2.VideoCapture(source,cv2.CAP_FFMPEG,[cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,int(cfg.get('capture_timeout_ms',5000)),cv2.CAP_PROP_READ_TIMEOUT_MSEC,int(cfg.get('capture_timeout_ms',5000))])
        else: cap=cv2.VideoCapture(source)
        if not cap.isOpened(): raise RuntimeError('Cannot open camera/video; source omitted for credential safety')
        h,w=map(int,cfg['size'])
        if isinstance(source,str) and Path(source).is_file(): result['source_sha256']=sha256(source)
        result['source_id']=cfg.get('source_id','unspecified')
        result['source_fps']=float(cap.get(cv2.CAP_PROP_FPS))
        result['result_kind']='sequential capture-read to decoded detections'
        result['camera_limitations']={'sensor_to_result_ms':None,'dropped_source_frames':None,'queue_policy':'sequential backend buffering; not bounded latest-frame service'}
    else: raise ValueError('mode must be model or camera')
    expected={'tensor(float)':np.dtype('float32'),'tensor(float16)':np.dtype('float16')}
    frame_index=-1
    def one(measure=False):
        begin=time.perf_counter_ns(); read_end=begin; pre_end=begin
        nonlocal x,frame_index
        frame_pts_ms=None
        if cap is not None:
            ok,frame=cap.read(); read_end=time.perf_counter_ns()
            if not ok: return None
            frame_index+=1
            frame_pts_ms=float(cap.get(cv2.CAP_PROP_POS_MSEC))
            x=preprocess_bgr(frame,h,w,cfg.get('mean',(.485,.456,.406)),cfg.get('std',(.229,.224,.225)))
            pre_end=time.perf_counter_ns()
        if inp.type not in expected or x.dtype!=expected[inp.type]: raise ValueError(f'Dtype mismatch: {inp.type} versus {x.dtype}; no implicit cast')
        if len(inp.shape)!=x.ndim or any(isinstance(d,int) and d!=a for d,a in zip(inp.shape,x.shape)): raise ValueError('Input shape differs from export')
        inf_begin=time.perf_counter_ns(); y=s.run(None,{inp.name:x}); inf_end=time.perf_counter_ns()
        n=0
        if cap is not None:
            boxes,scores,classes=decode(dict(zip(names,y)),cfg,frame.shape[:2]); n=len(scores)
        end=time.perf_counter_ns()
        return {'capture_ms':(read_end-begin)/1e6,'preprocess_ms':(pre_end-read_end)/1e6,
                'inference_ms':(inf_end-inf_begin)/1e6,'postprocess_ms':(end-inf_end)/1e6,
                'total_ms':(end-begin)/1e6,'frame_index':frame_index if cap is not None else None,'source_pts_ms':frame_pts_ms,'images':int(x.shape[0]),'detections':n}
    try:
        t=time.perf_counter(); first=one(); result['first_call_s']=time.perf_counter()-t
        if first is None: raise RuntimeError('Source ended before warmup')
        for _ in range(int(cfg.get('warmup',10))):
            if one() is None: raise RuntimeError('Source ended during warmup')
        rows=[]; start=time.perf_counter()
        for i in range(int(cfg.get('iterations',100))):
            row=one(True)
            if row is None: break
            row['iteration']=i; rows.append(row)
        elapsed=time.perf_counter()-start
    finally:
        if cap is not None: cap.release()
    if not rows: raise RuntimeError('No measured frames')
    if cfg.get('profile',False):
        result['profile_path']=s.end_profiling()
        result['operator_profile']=artifact(result['profile_path'])
    if len(rows)<int(cfg.get('iterations',100)) and not cfg.get('allow_short_video',False):
        raise RuntimeError('Source ended early; allow_short_video=true required for partial runs')
    with (out/'samples.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
    result.update({'status':'ok' if len(rows)==int(cfg.get('iterations',100)) else 'partial_source_eof',
                   'iterations':len(rows),'images':sum(r['images'] for r in rows),'elapsed_s':elapsed,
                   'throughput_fps':sum(r['images'] for r in rows)/elapsed,
                   'latency_ms':stats([r['total_ms'] for r in rows]),
                   'stages_ms':{k:stats([r[k] for r in rows]) for k in rows[0] if k.endswith('_ms') and k!='source_pts_ms'},
                   'timing_boundary':'host numpy input to host numpy output for inference; includes GPU transfers if GPU backend',
                   'profile_instrumented':bool(cfg.get('profile',False))})
    write_json(out/'result.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--profile',action='store_true');a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text())
    if a.profile: cfg['profile']=True
    r=run(cfg,a.output)
    print(json.dumps({k:r[k] for k in ('status','images','throughput_fps','latency_ms')},ensure_ascii=False))
