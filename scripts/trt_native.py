"""Static FP32/FP16 TensorRT 10/11 runtime adapter; no ORT provider/fallback.
GPU execution not tested in artifact authoring environment. Uses documented tensor-name API.
"""
from pathlib import Path
from types import SimpleNamespace
import numpy as np

class NativeSession:
    def __init__(self,cfg):
        import tensorrt as trt
        import torch
        if not torch.cuda.is_available():raise RuntimeError('Native TensorRT requires working CUDA torch and NVIDIA GPU')
        self.trt,self.torch=trt,torch
        self.device=int(cfg.get('device_id',0));torch.cuda.set_device(self.device)
        self.logger=trt.Logger(trt.Logger.WARNING);self.runtime=trt.Runtime(self.logger)
        self.engine=self.runtime.deserialize_cuda_engine(Path(cfg['engine']).read_bytes())
        if self.engine is None:raise RuntimeError('TensorRT engine deserialization failed; check GPU/version/provenance')
        self.context=self.engine.create_execution_context()
        if self.context is None:raise RuntimeError('TensorRT context creation failed')
        self.inputs=[];self.outputs=[];self.buffers={}
        dtypes={trt.float32:(np.dtype('float32'),torch.float32,'tensor(float)'),trt.float16:(np.dtype('float16'),torch.float16,'tensor(float16)')}
        self.numpy_dtypes={}
        for i in range(self.engine.num_io_tensors):
            name=self.engine.get_tensor_name(i);shape=tuple(self.engine.get_tensor_shape(name));dtype=self.engine.get_tensor_dtype(name)
            if dtype not in dtypes:raise ValueError(f'Unsupported detection I/O dtype: {dtype}')
            if not shape or min(shape)<1:raise ValueError('Native baseline requires fixed positive I/O shapes; export static batch1')
            if self.engine.get_tensor_location(name)!=trt.TensorLocation.DEVICE:raise ValueError('Host/shape I/O not supported in static detection adapter')
            nd,td,tag=dtypes[dtype];self.numpy_dtypes[name]=nd
            info=SimpleNamespace(name=name,shape=list(shape),type=tag)
            (self.inputs if self.engine.get_tensor_mode(name)==trt.TensorIOMode.INPUT else self.outputs).append(info)
            self.buffers[name]=torch.empty(shape,dtype=td,device=f'cuda:{self.device}')
            if not self.context.set_tensor_address(name,self.buffers[name].data_ptr()):raise RuntimeError('TensorRT set_tensor_address failed')
        if len(self.inputs)!=1 or len(self.inputs[0].shape)!=4:raise ValueError('One NCHW image input required')
        self.stream=torch.cuda.Stream(device=self.device)
        torch.cuda.synchronize(self.device)
    def get_inputs(self):return self.inputs
    def get_outputs(self):return self.outputs
    def get_providers(self):return ['TensorRTNative']
    def run(self,output_names,feed):
        torch=self.torch;torch.cuda.set_device(self.device);inp=self.inputs[0]
        if set(feed)!={inp.name}:raise ValueError('Wrong engine input names')
        x=feed[inp.name]
        if list(x.shape)!=inp.shape or x.dtype!=self.numpy_dtypes[inp.name]:raise ValueError('Engine input shape/dtype mismatch')
        names=[o.name for o in self.outputs] if output_names is None else output_names
        if any(n not in [o.name for o in self.outputs] for n in names):raise ValueError('Unknown engine output')
        # Host-call boundary includes H2D + enqueue + completion + D2H and wrapper overhead.
        with torch.cuda.stream(self.stream):
            self.buffers[inp.name].copy_(torch.from_numpy(np.ascontiguousarray(x)))
            if not self.context.execute_async_v3(stream_handle=self.stream.cuda_stream):raise RuntimeError('TensorRT execution failed')
            result=[self.buffers[n].cpu().numpy() for n in names]
        self.stream.synchronize()
        return result
