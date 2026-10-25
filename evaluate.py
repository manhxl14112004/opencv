"""Held-out original and edge-truncation evaluation. Does not change model."""
import json,time
from pathlib import Path
import torch
from PIL import Image
from model import SortNet,preprocess,CLASSES,decision
from foreground import locate_product
from dataset_paths import load_manifest

base=Path(__file__).parents[1];torch.set_num_threads(4)
checkpoint=torch.load(base/'models/sortnet.pt',map_location='cpu',weights_only=True)
model=SortNet();model.load_state_dict(checkpoint['state_dict']);model.eval();transform=preprocess()
rows=load_manifest(base/'models/dataset_manifest.json')['images']
results=[]
with torch.inference_mode():
    for row in rows:
        if row['split']!='test':continue
        with Image.open(row['path']) as original:
            image=original.convert('RGB');w,h=image.size
            variants={'original':image,'left_20':image.crop((int(.2*w),0,w,h)),
                      'right_20':image.crop((0,0,int(.8*w),h)),
                      'top_20':image.crop((0,int(.2*h),w,h)),
                      'bottom_20':image.crop((0,0,w,int(.8*h)))}
            for name,picture in variants.items():
                start=time.perf_counter()
                located=locate_product(picture,checkpoint['foreground_config']) if checkpoint.get('feature_transform')=='foreground-v1' else None
                if checkpoint.get('feature_transform')=='foreground-v1' and located is None:
                    d={'class_code':3,'product_type':'unknown','confidence':0.,'rejection_reasons':['no_product_in_roi']}
                else:
                    embedding=model.embedding(transform(located.image if located else picture).unsqueeze(0));prob=model.head(embedding).softmax(1)[0]
                    d=decision(prob,embedding[0],checkpoint['centroids'],checkpoint['distance_limits'])
                d.update(variant=name,true_label=row['label'],source=Path(row['path']).name,total_preprocess_inference_ms=(time.perf_counter()-start)*1000)
                results.append(d)
report={}
for variant in ['original','left_20','right_20','top_20','bottom_20']:
    selected=[r for r in results if r['variant']==variant]
    report[variant]={'n':len(selected),'correct':sum(r['product_type']==r['true_label'] for r in selected),
                     'unknown':sum(r['class_code']==3 for r in selected),
                     'wrong_accepted':sum(r['class_code']!=3 and r['product_type']!=r['true_label'] for r in selected)}
output={'model_version':checkpoint['version'],'summary':report,'limitations':'Synthetic crops of held-out images from the supplied setup; not an independent conveyor or real occlusion benchmark. No unknown ground-truth set.', 'predictions':results}
(base/'models/occlusion_evaluation.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
