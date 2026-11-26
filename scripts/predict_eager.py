#!/usr/bin/env python3
"""Independent FP32 eager COCO predictions using torch preprocessing and decoding."""
import argparse,json
from pathlib import Path
from PIL import Image
from train import load_model,versions,sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-config',required=True);p.add_argument('--decode-config',required=True)
    p.add_argument('--annotations',required=True);p.add_argument('--images',required=True)
    p.add_argument('--image-ids');p.add_argument('--output-dir',required=True);a=p.parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F
    mc=json.loads(Path(a.model_config).read_text());dc=json.loads(Path(a.decode_config).read_text())
    for key in ('size','background_index','category_ids'): 
        if key not in dc: raise ValueError(f'Explicit {key} required')
    package_versions=versions(mc.get('expected_versions'))
    model=load_model(mc,'cpu');module=model.model.model.cpu().float().eval()
    gt=json.loads(Path(a.annotations).read_text());ids=set(json.loads(Path(a.image_ids).read_text())) if a.image_ids else {x['id'] for x in gt['images']}
    if not ids <= {x['id'] for x in gt['images']}: raise ValueError('Unknown test image ID')
    root=Path(a.images).resolve();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
    mean=torch.tensor(dc.get('mean',[.485,.456,.406])).reshape(1,3,1,1)
    std=torch.tensor(dc.get('std',[.229,.224,.225])).reshape(1,3,1,1)
    rows=[];input_files=[]
    with torch.inference_mode():
        for record in sorted(gt['images'],key=lambda x:x['id']):
            if record['id'] not in ids:continue
            path=(root/record['file_name']).resolve()
            if not path.is_relative_to(root):raise ValueError('Image escapes dataset root')
            rgb=np.asarray(Image.open(path).convert('RGB')).copy();h,w=rgb.shape[:2]
            if (w,h)!=(record['width'],record['height']):raise ValueError('Annotation geometry differs')
            x=torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float()/255.
            x=(F.interpolate(x,size=tuple(dc['size']),mode='bilinear',align_corners=False,antialias=False)-mean)/std
            y=module(x)
            if not isinstance(y,dict) or not {'pred_boxes','pred_logits'}<=y.keys():raise RuntimeError('Eager internal API changed')
            boxes=y['pred_boxes'][0];logits=y['pred_logits'][0]
            if not torch.isfinite(boxes).all() or not torch.isfinite(logits).all():raise ValueError('Nonfinite output')
            scores=logits.sigmoid();slots=torch.arange(scores.shape[1]);bg=dc['background_index']
            if bg is not None:
                if not -len(slots)<=int(bg)<len(slots):raise ValueError('Invalid background')
                keep=slots!=slots[int(bg)];scores=scores[:,keep];slots=slots[keep]
            flat=scores.flatten();k=min(int(dc.get('accuracy_top_k',dc.get('top_k',300))),len(flat))
            selected=torch.argsort(flat,descending=True,stable=True)[:k]
            selected=selected[flat[selected]>float(dc.get('accuracy_threshold',.001))]
            q=selected//scores.shape[1];classes=slots[selected%scores.shape[1]]
            b=boxes[q];xyxy=torch.cat((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2),1)*torch.tensor([w,h,w,h])
            xyxy=torch.minimum(torch.maximum(xyxy,torch.tensor(0.)),torch.tensor([w,h,w,h]))
            for box,score,cls in zip(xyxy.tolist(),flat[selected].tolist(),classes.tolist()):
                mapping=dc['category_ids']
                if isinstance(mapping,dict):
                    if str(cls) not in mapping: raise ValueError(f'Missing mapping for raw class slot {cls}')
                    category=mapping[str(cls)]
                else: category=mapping[cls]
                if category is None: continue
                x0,y0,x1,y1=box
                rows.append({'image_id':record['id'],'category_id':int(category),'bbox':[x0,y0,x1-x0,y1-y0],'score':score})
            input_files.append({'file':record['file_name'],'sha256':sha256(path)})
    result=out/'predictions.coco.json';result.write_text(json.dumps(rows))
    (out/'manifest.json').write_text(json.dumps({'model_config':mc,'decode_config':dc,'versions':package_versions,'images':input_files,
        'checkpoint_sha256':sha256(mc['model']['kwargs']['pretrain_weights']),'annotation_sha256':sha256(a.annotations),'predictions_sha256':sha256(result),
        'adapter':'model.model.model eager; torch FP32 interpolate/sigmoid/sort/decode independently from NumPy adapter; fail-fast on internal API change'},indent=2))
    print(result)

if __name__=='__main__':main()
