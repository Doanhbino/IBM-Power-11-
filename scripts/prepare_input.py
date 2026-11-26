#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from common import preprocess_bgr
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--image',required=True);p.add_argument('--output',required=True);p.add_argument('--input-name',required=True);a=p.parse_args()
c=json.loads(Path(a.config).read_text());h,w=c['size'];rgb=np.asarray(Image.open(a.image).convert('RGB'))
x=preprocess_bgr(rgb[:,:,::-1],h,w,c.get('mean',(.485,.456,.406)),c.get('std',(.229,.224,.225)))
np.savez(a.output,**{a.input_name:x})
