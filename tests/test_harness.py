"""Small ONNX fixture tests harness correctness, not RF-DETR accuracy/performance."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h,TensorProto,numpy_helper
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

class Harness(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
        x=np.zeros((1,3,8,8),np.float32)
        boxes=np.full((1,2,4),.5,np.float32)
        labels=np.array([[[5,-5,-10],[4,-4,-10]]],np.float32)
        nodes=[h.make_node('ReduceMean',['input'],['mean'],keepdims=0),h.make_node('Mul',['mean','zero'],['z']),h.make_node('Add',['boxes','z'],['dets']),h.make_node('Add',['logits','z'],['labels'])]
        graph=h.make_graph(nodes,'harness-fixture',[h.make_tensor_value_info('input',TensorProto.FLOAT,[1,3,8,8])],[h.make_tensor_value_info('dets',TensorProto.FLOAT,[1,2,4]),h.make_tensor_value_info('labels',TensorProto.FLOAT,[1,2,3])],[numpy_helper.from_array(boxes,'boxes'),numpy_helper.from_array(labels,'logits'),numpy_helper.from_array(np.array(0,np.float32),'zero')])
        m=h.make_model(graph,opset_imports=[h.make_opsetid('',17)]);m.ir_version=9;onnx.save(m,self.p/'model.onnx')
        np.savez(self.p/'input.npz',input=x);np.savez(self.p/'ref.npz',input_0=x,dets_0=boxes,labels_0=labels)
        self.cfg={'model':str(self.p/'model.onnx'),'input_npz':str(self.p/'input.npz'),'mode':'model','threads':1,'warmup':1,'iterations':2,'size':[8,8],'boxes_output':'dets','logits_output':'labels','background_index':-1,'category_ids':{'0':1,'1':2}}
        self.c=self.p/'cfg.json';self.c.write_text(json.dumps(self.cfg))
    def tearDown(self):self.tmp.cleanup()
    def cli(self,script,*args,code=0):
        r=subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],capture_output=True,text=True)
        self.assertEqual(r.returncode,code,r.stdout+r.stderr);return r
    def test_model_profile_and_report(self):
        self.cli('benchmark.py','--config',self.c,'--output',self.p/'run','--profile')
        r=json.loads((self.p/'run/result.json').read_text());self.assertEqual(r['images'],2);self.assertTrue(r['profile_instrumented'])
        self.cli('profile_ops.py',r['profile_path'],'--output',self.p/'ops.csv')
        self.assertIn('CPUExecutionProvider',(self.p/'ops.csv').read_text())
        self.cli('report.py',self.p/'run','--output',self.p/'report');self.assertTrue((self.p/'report/report.csv').exists())
    def test_parity_pass_and_failure(self):
        self.cli('validate_onnx.py','--model',self.p/'model.onnx','--reference',self.p/'ref.npz','--output',self.p/'pass.json')
        z=dict(np.load(self.p/'ref.npz'));z['labels_0']+=1;np.savez(self.p/'bad.npz',**z)
        self.cli('validate_onnx.py','--model',self.p/'model.onnx','--reference',self.p/'bad.npz','--output',self.p/'fail.json',code=2)
    def test_camera_and_coco(self):
        import cv2
        v=cv2.VideoWriter(str(self.p/'replay.avi'),cv2.VideoWriter_fourcc(*'MJPG'),10,(16,16))
        self.assertTrue(v.isOpened())
        for _ in range(8):v.write(np.zeros((16,16,3),np.uint8))
        v.release();c=dict(self.cfg,mode='camera',source=str(self.p/'replay.avi'));self.c.write_text(json.dumps(c))
        self.cli('benchmark.py','--config',self.c,'--output',self.p/'camera')
        r=json.loads((self.p/'camera/result.json').read_text());self.assertEqual(r['images'],2)
        cv2.imwrite(str(self.p/'image.png'),np.zeros((16,16,3),np.uint8))
        gt={'images':[{'id':1,'file_name':'image.png','width':16,'height':16}],'categories':[{'id':1,'name':'one'},{'id':2,'name':'two'}],'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[4,4,8,8],'area':64,'iscrowd':0}]}
        (self.p/'gt.json').write_text(json.dumps(gt))
        self.cli('evaluate_coco.py','--config',self.c,'--annotations',self.p/'gt.json','--images',self.p,'--output-dir',self.p/'accuracy')
        self.assertTrue((self.p/'accuracy/accuracy.json').exists())
    def test_external_hash(self):
        from common import model_files
        m=onnx.load(self.p/'model.onnx');onnx.save_model(m,str(self.p/'external.onnx'),save_as_external_data=True,all_tensors_to_one_file=True,location='weights.bin',size_threshold=0)
        files=model_files(self.p/'external.onnx');self.assertEqual(len(files),2)

    def test_real_profile_shapes_to_cpp_microbenchmark(self):
        if not (ROOT/'build/rfdetr_gemm').is_file():
            self.skipTest('Build the C++ core first')
        graph=h.make_graph([h.make_node('MatMul',['input','weight'],['output'],name='fixture_matmul')],
            'profile-shape-fixture',[h.make_tensor_value_info('input',TensorProto.FLOAT,[1,3,8,8])],
            [h.make_tensor_value_info('output',TensorProto.FLOAT,[1,3,8,8])],
            [numpy_helper.from_array(np.eye(8,dtype=np.float32),'weight')])
        model=h.make_model(graph,opset_imports=[h.make_opsetid('',17)]);model.ir_version=9
        onnx.save(model,self.p/'matmul.onnx')
        cfg=dict(self.cfg,model=str(self.p/'matmul.onnx'));self.c.write_text(json.dumps(cfg))
        self.cli('benchmark.py','--config',self.c,'--output',self.p/'profile','--profile')
        result=json.loads((self.p/'profile/result.json').read_text())
        self.cli('profile_ops.py',result['profile_path'],'--output',self.p/'ops.csv','--shapes',self.p/'shapes.json',
                 '--model',result['optimized_model']['path'])
        cases=json.loads((self.p/'shapes.json').read_text())['cases']
        self.assertEqual((cases[0]['m'],cases[0]['n'],cases[0]['k']),(8,8,8))
        self.cli('gemm_sweep.py','--shapes',self.p/'shapes.json','--output',self.p/'gemm',
                 '--backends','reference','--iterations','1')
        report=json.loads((self.p/'gemm/run.json').read_text())
        self.assertEqual(report['status'],'ok');self.assertEqual(report['results'][0]['repeats'],3)

    def test_concurrent_workers(self):
        cfg={'base_config':str(self.c),'threads':[1],'smt_limits':[1],'layouts':['all'],'workers':[2],'repeats':1}
        sweep=self.p/'sweep.json';sweep.write_text(json.dumps(cfg))
        self.cli('sweep.py','--config',sweep,'--output',self.p/'workers')
        rows=json.loads((self.p/'workers/results.json').read_text())
        if not rows:self.skipTest('Fewer than two allowed CPU cores')
        self.assertTrue(rows[0]['valid'])
        self.assertEqual(rows[0]['images'],4)
        self.assertFalse(set(rows[0]['cpu_masks'][0]) & set(rows[0]['cpu_masks'][1]))
        self.assertAlmostEqual(rows[0]['throughput_makespan_fps'],4/rows[0]['makespan_s'])

if __name__=='__main__':unittest.main()
