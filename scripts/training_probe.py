#!/usr/bin/env python3
"""Optional version-guarded instrumentation of RF-DETR's existing training loop."""
import argparse,csv,inspect,json,os,sys,time
from pathlib import Path
from unittest.mock import patch
from run_record import artifact, atomic_json


def run():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--sync-cuda',action='store_true',help='Synchronize CUDA at batch boundaries; changes throughput')
    p.add_argument('--profile',action='store_true',help='Short torch.profiler capture, separate from final KPI')
    p.add_argument('--wait',type=int,default=1);p.add_argument('--warmup',type=int,default=1);p.add_argument('--active',type=int,default=3)
    a=p.parse_args()
    if min(a.wait,a.warmup)<0 or a.active<1:p.error('Invalid profiler schedule')
    cfg=json.loads(Path(a.config).read_text());output=Path(cfg['train']['output_dir'])
    if output.exists() and any(output.iterdir()):raise RuntimeError('Use a new output_dir for this instrumented run')
    if cfg['train']['device']=='cuda' and os.environ.get('NVIDIA_TF32_OVERRIDE')!='0':
        raise RuntimeError('Launch with NVIDIA_TF32_OVERRIDE=0 for strict FP32')
    import torch
    import rfdetr.training as training
    from pytorch_lightning.callbacks import Callback
    from train import main as train_main
    factory=training.build_trainer
    if 'extra_callbacks' not in inspect.signature(factory).parameters:
        raise RuntimeError('Unsupported RF-DETR factory: extra_callbacks absent; pin/adapt version')
    target=output/'probe'

    class Probe(Callback):
        def __init__(self):
            super().__init__();self.batches=[];self.epochs=[];self.prof=None;self.count=0;self.epoch_start=0.;self.status='created'
        def sync(self,module):
            if a.sync_cuda and module.device.type=='cuda':torch.cuda.synchronize(module.device)
        def write(self):
            target.mkdir(parents=True,exist_ok=True)
            for name,rows in [('batches',self.batches),('epochs',self.epochs)]:
                if rows:
                    with (target/(name+'.csv')).open('w',newline='') as f:
                        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            (target/'summary.json').write_text(json.dumps({'status':self.status,'instrumented':True,'sync_cuda':a.sync_cuda,
                'profile':a.profile,'batch_scope':'PTL on_train_batch_start to end: after transfer; includes forward/backward and optimizer on update batches; excludes dataloader wait',
                'epoch_scope':'PTL train epoch callbacks, may include validation/checkpoint depending loop; not pure compute',
                'throughput_warning':'CUDA host times may be asynchronous unless sync_cuda; instrumentation overhead invalidates final KPI',
                'epochs':self.epochs,'batch_count':len(self.batches)},indent=2))
        def stop_profile(self):
            if self.prof is not None:
                self.prof.stop()
                (target/'operators.txt').write_text(self.prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=100))
                self.prof=None
        def on_fit_start(self,trainer,module):
            if trainer.world_size!=1:raise RuntimeError('Probe supports single-process training only; distributed metrics need rank aggregation')
            target.mkdir(parents=True,exist_ok=True);self.status='running'
            if a.profile:
                activities=[torch.profiler.ProfilerActivity.CPU]
                if module.device.type=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
                def trace(prof):prof.export_chrome_trace(str(target/('trace-step-%d.json'%prof.step_num)))
                self.prof=torch.profiler.profile(activities=activities,schedule=torch.profiler.schedule(wait=a.wait,warmup=a.warmup,active=a.active,repeat=1),on_trace_ready=trace,record_shapes=True)
                self.prof.start()
        def on_train_epoch_start(self,trainer,module):
            self.sync(module);self.epoch_start=time.perf_counter();self.epoch_offset=len(self.batches)
        def on_train_batch_start(self,trainer,module,batch,batch_idx):
            self.sync(module);self.start=time.perf_counter();self.images=None
            try:
                samples=batch[0];tensor=samples.tensors if hasattr(samples,'tensors') else samples
                self.images=int(tensor.shape[0])
            except (AttributeError,IndexError,KeyError,TypeError):pass
        def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
            self.sync(module);seconds=time.perf_counter()-self.start
            loss = outputs.get('loss') if isinstance(outputs,dict) else outputs
            if isinstance(loss,torch.Tensor) and not torch.isfinite(loss.detach()).all():
                raise RuntimeError('Nonfinite training loss; smoke gate failed')
            self.batches.append({'epoch':int(trainer.current_epoch),'batch_idx':batch_idx,'global_optimizer_step':int(trainer.global_step),
                'images':self.images,'host_batch_s':seconds,'peak_rss_kib_linux':__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss})
            self.count+=1
            if self.prof is not None:
                self.prof.step()
                if self.count>=a.wait+a.warmup+a.active:self.stop_profile()
        def on_train_epoch_end(self,trainer,module):
            self.sync(module);elapsed=time.perf_counter()-self.epoch_start;rows=self.batches[self.epoch_offset:]
            images=sum(x['images'] for x in rows) if rows and all(x['images'] is not None for x in rows) else None
            self.epochs.append({'epoch':int(trainer.current_epoch),'epoch_callback_wall_s':elapsed,'batches':len(rows),'images':images,
                'images_per_epoch_wall_s':images/elapsed if images is not None and elapsed>0 else None})
            self.write()
        def on_fit_end(self,trainer,module):
            self.stop_profile();self.status='completed';self.write()
        def on_exception(self,trainer,module,exception):
            self.status='failed';self.stop_profile();self.write()

    probe=Probe()
    def wrapped(*args,**kwargs):
        kwargs['extra_callbacks']=[*list(kwargs.get('extra_callbacks',()) or ()),probe]
        return factory(*args,**kwargs)
    # High-level RFDETR.train imports this factory at call time in the verified API.
    with patch.object(training,'build_trainer',new=wrapped),patch.object(sys,'argv',['train.py','--config',a.config]):
        train_main(profile_instrumented=a.profile)
    if not probe.batches:raise RuntimeError('Factory instrumentation did not observe training batches; do not report probe performance')
    if a.profile:
        profiles=sorted(target.glob('trace-step-*.json'))
        if not profiles:raise RuntimeError('No training trace captured; increase steps or shorten profiler schedule')
        manifest_path=output/'baseline_manifest.json'
        manifest=json.loads(manifest_path.read_text())
        manifest['operator_profile']=artifact(profiles[0])
        manifest['operator_profiles']=[artifact(path) for path in profiles]
        atomic_json(manifest_path,manifest)
    print(target)

if __name__=='__main__':run()
