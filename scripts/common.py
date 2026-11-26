"""Small explicit RF-DETR detection adapter; no torch/rfdetr dependency at inference."""
import importlib.metadata
import os
import platform
from pathlib import Path

import numpy as np
from run_record import atomic_json, public_config, sha256


def session(cfg, profile=False):
    if cfg.get('backend')=='tensorrt_native':
        if profile: raise ValueError('Use trtexec/Nsight for native TensorRT profile, separate from KPI')
        from trt_native import NativeSession
        return NativeSession(cfg)
    import onnxruntime as ort
    so = ort.SessionOptions()
    threads = int(cfg.get('threads', 1))
    if threads < 1: raise ValueError('threads must be >= 1')
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = int(cfg.get('inter_threads', 1))
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.add_session_config_entry('session.intra_op.allow_spinning', str(int(cfg.get('spinning', False))))
    so.enable_profiling = profile
    so.profile_file_prefix = cfg.get('profile_prefix', 'ort-profile')
    if profile and cfg.get('backend', 'cpu') == 'cpu':
        so.optimized_model_filepath = str(Path(so.profile_file_prefix).parent / 'optimized.onnx')
    backend = cfg.get('backend', 'cpu')
    if backend == 'cpu': providers = ['CPUExecutionProvider']
    elif backend == 'tensorrt':
        if 'TensorrtExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('TensorRT EP unavailable; refusing silent CPU comparison')
        providers = [('TensorrtExecutionProvider', cfg.get('provider_options', {})), 'CUDAExecutionProvider', 'CPUExecutionProvider']
    else: raise ValueError('backend must be cpu or tensorrt')
    s = ort.InferenceSession(cfg['model'], sess_options=so, providers=providers)
    if backend == 'tensorrt' and 'TensorrtExecutionProvider' not in s.get_providers():
        raise RuntimeError('TensorRT EP failed initialization; refusing CPU-only fallback result')
    if len(s.get_inputs()) != 1: raise ValueError('Detection baseline requires one NCHW image input')
    return s


def preprocess_bgr(frame, h, w, mean=(.485,.456,.406), std=(.229,.224,.225)):
    # Tensor bilinear, align_corners=False, antialias=False (half-pixel coordinates).
    # NumPy reference implementation; optimize later under eager preprocessing parity gate.
    rgb=frame[:,:,::-1].astype(np.float32)/np.float32(255.)
    ih,iw=rgb.shape[:2]
    yy=np.maximum((np.arange(h,dtype=np.float64)+.5)*ih/h-.5,0)
    xx=np.maximum((np.arange(w,dtype=np.float64)+.5)*iw/w-.5,0)
    y0=np.floor(yy).astype(int);x0=np.floor(xx).astype(int)
    y1=np.minimum(y0+1,ih-1);x1=np.minimum(x0+1,iw-1)
    wy=(yy-y0).astype(np.float32)[:,None,None]
    wx=(xx-x0).astype(np.float32)[None,:,None]
    top=rgb[y0[:,None],x0[None,:]]*(1-wx)+rgb[y0[:,None],x1[None,:]]*wx
    bottom=rgb[y1[:,None],x0[None,:]]*(1-wx)+rgb[y1[:,None],x1[None,:]]*wx
    resized=top*(1-wy)+bottom*wy
    resized=(resized-np.asarray(mean,dtype=np.float32))/np.asarray(std,dtype=np.float32)
    return np.ascontiguousarray(resized.transpose(2,0,1)[None])


def decode(outputs, cfg, original_hw):
    """Batch=1 detection; normalized cxcywh + sigmoid logits. Returns RAW class slots."""
    for key in ('boxes_output','logits_output','background_index'):
        if key not in cfg: raise ValueError(f'Explicit {key} required; inspect model, never infer by shape')
    boxes = np.asarray(outputs[cfg['boxes_output']])
    logits = np.asarray(outputs[cfg['logits_output']])
    if boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[2] != 4:
        raise ValueError('Expected boxes [1,Q,4] normalized cxcywh')
    if logits.ndim != 3 or logits.shape[:2] != boxes.shape[:2]: raise ValueError('Expected logits [1,Q,C]')
    if not np.all(np.isfinite(boxes)) or not np.all(np.isfinite(logits)): raise ValueError('Nonfinite model output')
    if not np.issubdtype(logits.dtype,np.floating): raise ValueError('Floating logits required')
    work=logits[0].astype(np.result_type(logits.dtype,np.float32))
    scores=(1. / (1. + np.exp(-np.clip(work,-88,88)))).astype(logits.dtype)
    slots = np.arange(scores.shape[1])
    bg = cfg['background_index']
    if bg is not None:
        bg = int(bg)
        if not -len(slots) <= bg < len(slots): raise ValueError('Invalid background_index')
        keep = slots != slots[bg]
        scores, slots = scores[:, keep], slots[keep]
    flat = scores.ravel()
    k = min(int(cfg.get('top_k',logits.shape[1])), len(flat))
    if k < 1: raise ValueError('top_k/classes must be positive')
    # Stable full ordering is intentional reference baseline; optimize only with parity check.
    ids = np.argsort(-flat, kind='stable')[:k]
    ids = ids[flat[ids] > float(cfg.get('threshold',.25))]
    q, c = np.unravel_index(ids, scores.shape)
    b = boxes[0,q]
    xyxy = np.concatenate((b[:,:2]-b[:,2:]/2, b[:,:2]+b[:,2:]/2),axis=1)
    h,w = original_hw
    xyxy *= np.asarray([w,h,w,h],dtype=np.float32)
    xyxy[:,[0,2]] = np.clip(xyxy[:,[0,2]],0,w)
    xyxy[:,[1,3]] = np.clip(xyxy[:,[1,3]],0,h)
    return xyxy, flat[ids], slots[c]


def model_files(path):
    import onnx
    p=Path(path).resolve()
    model=onnx.load(str(p),load_external_data=False)
    locations=set()
    def visit(message):
        if message.DESCRIPTOR.full_name=='onnx.TensorProto' and message.data_location==onnx.TensorProto.EXTERNAL:
            for entry in message.external_data:
                if entry.key=='location': locations.add(entry.value)
        for field,value in message.ListFields():
            if field.message_type is not None:
                if field.is_repeated:
                    for item in value: visit(item)
                else: visit(value)
    visit(model)
    return [{'file':str(p),'sha256':sha256(p)}]+[{'file':name,'sha256':sha256(p.parent/name)} for name in sorted(locations)]


def manifest(cfg):
    packages = {}
    for name in ('numpy','onnx','onnxruntime','onnxruntime-gpu','Pillow','opencv-python','opencv-python-headless','torch','torchvision','rfdetr','tensorrt'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    c = public_config(cfg)
    return {'schema_version':1,'platform':platform.platform(),'machine':platform.machine(),
            'python':platform.python_version(),'packages':packages,'config':c,
            'affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            'model_sha256':sha256(cfg['model']),'model_files':model_files(cfg['model']),
            'engine_sha256':sha256(cfg['engine']) if cfg.get('engine') else None,
            'backend_scope':'Native TensorRT; no ORT fallback; source ONNX to engine provenance must match build logs' if cfg.get('backend')=='tensorrt_native' else 'ORT TensorRT with CUDA/CPU fallback; provider assignment needs profile audit' if cfg.get('backend')=='tensorrt' else 'ORT CPUExecutionProvider; MMA kernel usage unproven'}


def stats(values):
    a=np.asarray(values,dtype=float)
    if not len(a): return {}
    return {'mean':float(a.mean()),'p50':float(np.percentile(a,50)),
            'p95':float(np.percentile(a,95)),'p99':float(np.percentile(a,99)), 'max':float(a.max())}


def write_json(path, obj):
    atomic_json(path, obj)
