"""Compare saved models on fixed, visually reviewed development-camera files."""
import argparse,json,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from PIL import Image
from model import SortNet,preprocess,decision
from foreground import locate_product
from dataset_paths import load_manifest


def predict(checkpoint_path,rows):
    saved=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    model=SortNet();model.load_state_dict(saved['state_dict']);model.eval()
    transform=preprocess(False)
    with torch.inference_mode():
        for _ in range(8):model(torch.zeros(1,3,224,224))
        results=[]
        for row in rows:
            with Image.open(row['path']) as original:
                rgb=original.convert('RGB');start=time.perf_counter()
                located=locate_product(rgb,saved['foreground_config'])
                if located is None:
                    result={'product_type':'unknown','class_code':3,'confidence':0.,'rejection_reasons':['no_product_in_roi']}
                else:
                    embedding=model.embedding(transform(located.image).unsqueeze(0))
                    probability=model.head(embedding).softmax(1)[0]
                    result=decision(probability,embedding[0],saved['centroids'],saved['distance_limits'])
                result.update(file=Path(row['path']).name,true_label=row['label'],split=row['split'],group=row['group'],
                              bbox=list(located.bbox) if located else None,total_ms=(time.perf_counter()-start)*1000)
                results.append(result)
    groups={}
    for group in sorted({r['group'] for r in results}):
        selected=[r for r in results if r['group']==group]
        groups[group]={'count':len(selected),'correct':sum(r['product_type']==r['true_label'] for r in selected),
                       'unknown':sum(r['class_code']==3 for r in selected),
                       'wrong_accepted':sum(r['class_code']!=3 and r['product_type']!=r['true_label'] for r in selected),
                       'errors':[r['file'] for r in selected if r['product_type']!=r['true_label']]}
    times=[r['total_ms'] for r in results]
    return {'model_version':saved['version'],'groups':groups,'total_ms_p50':float(np.percentile(times,50)),
            'total_ms_p95':float(np.percentile(times,95)),'predictions':results}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--previous-audit',type=Path,required=True)
    p.add_argument('--captures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    rows=[]
    for row in load_manifest(a.manifest)['images']:
        if row.get('source')=='night_camera':rows.append({**row,'group':'night_'+row['split']})
        elif row['split']=='test':rows.append({**row,'group':'original_day_test'})
    for row in json.loads(a.previous_audit.read_text(encoding='utf-8'))['results']:
        if not row['used_for_camera_calibration']:
            rows.append({'path':str(a.captures/row['file']),'label':row['true_label'],'split':'audit',
                         'group':'day_unknown_regression' if row['true_label']=='unknown' else 'day_camera_regression'})
    result={'limitations':'Fixed development files from the same physical products/setup. Night test files were excluded from model fitting; they are not an independent installation/session test. Unknown regression is only six previously reviewed files.',
            'baseline':predict(a.baseline,rows),'candidate':predict(a.candidate,rows)}
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for key in ['baseline','candidate']:
        print(key,result[key]['model_version'],json.dumps(result[key]['groups'],ensure_ascii=False),flush=True)
