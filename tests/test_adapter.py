import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from common import decode,preprocess_bgr

class AdapterTests(unittest.TestCase):
    def test_rgb_normalization(self):
        bgr=np.asarray([[[0,0,255]]],dtype=np.uint8)
        x=preprocess_bgr(bgr,1,1,mean=(0,0,0),std=(1,1,1))
        np.testing.assert_equal(x,np.array([[[[1]],[[0]],[[0]]]],dtype=np.float32))
    def test_half_pixel_resize(self):
        a=np.zeros((2,2,3),dtype=np.uint8);a[0,1]=100;a[1,0]=200;a[1,1]=100
        x=preprocess_bgr(a,1,1,mean=(0,0,0),std=(1,1,1))
        np.testing.assert_allclose(x,np.full((1,3,1,1),100/255,dtype=np.float32))
    def test_multilabel_and_background(self):
        d={'dets':np.array([[[.5,.5,.5,.5]]]),'labels':np.array([[[5.,4.,10.]]])}
        c={'boxes_output':'dets','logits_output':'labels','background_index':-1,'top_k':2,'threshold':.5}
        b,s,k=decode(d,c,(100,200))
        self.assertEqual(list(k),[0,1]);np.testing.assert_equal(b,[[50,25,150,75]]*2)
    def test_three_classes_collision(self):
        d={'dets':np.ones((1,2,4))*.5,'labels':np.zeros((1,2,4))}
        b,s,k=decode(d,{'boxes_output':'dets','logits_output':'labels','background_index':-1,'top_k':2},(20,30))
        self.assertEqual(len(s),2)
    def test_requires_background(self):
        with self.assertRaises(ValueError):decode({}, {},(1,1))

    def test_trt_initialization_fallback_rejected(self):
        from unittest.mock import patch,Mock
        from common import session
        fake=Mock();fake.get_providers.return_value=['CPUExecutionProvider']
        with patch('onnxruntime.get_available_providers',return_value=['TensorrtExecutionProvider','CPUExecutionProvider']),patch('onnxruntime.InferenceSession',return_value=fake):
            with self.assertRaises(RuntimeError):session({'model':'unused.onnx','backend':'tensorrt','threads':1})

if __name__=='__main__':unittest.main()
