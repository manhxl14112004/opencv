from __future__ import annotations
import argparse, copy, json, random, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch import nn
from model import SortNet, CLASSES, preprocess, decision
from foreground import locate_product, DEFAULTS as FOREGROUND_DEFAULTS
from dataset_paths import load_manifest


def train(manifest: Path, output: Path, epochs=160, views=6, threads=4, foreground=True, low_light=False, foreground_config=None):
    torch.set_num_threads(threads)
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    data=load_manifest(manifest)
    bundled_hub=Path(__file__).resolve().parents[1]/'models/torch_cache/hub'
    if (bundled_hub/'checkpoints/mobilenet_v3_small-047dcff4.pth').is_file():
        torch.hub.set_dir(str(bundled_hub))
    model=SortNet(pretrained=True).eval()
    for p in model.features.parameters(): p.requires_grad=False
    foreground_config=foreground_config or FOREGROUND_DEFAULTS
    transforms={False:preprocess(False,foreground),True:preprocess(True,foreground,low_light)}
    def features(split, augmented=False):
        xs,ys=[],[]
        selected=[r for r in data['images'] if r['split']==split]
        images=[]
        for row in selected:
            with Image.open(row['path']) as image:
                rgb=image.convert('RGB')
                if foreground:
                    located=locate_product(rgb,foreground_config)
                    if located is None:raise ValueError('No product located in training image: '+row['path'])
                    rgb=located.image
                for _ in range(views if augmented else 1):
                    images.append((transforms[augmented](rgb),CLASSES.index(row['label'])))
        with torch.inference_mode():
            for i in range(0,len(images),16):
                batch=images[i:i+16]
                xs.append(model.embedding(torch.stack([x for x,_ in batch])).clone())
                ys.extend(y for _,y in batch)
        return torch.cat(xs).clone(),torch.tensor(ys)
    started=time.perf_counter()
    xtrain,ytrain=features('train',True)
    xval,yval=features('val')
    xtest,ytest=features('test')
    xclean,yclean=features('train')
    # Convert inference tensors before using them in autograd.
    xtrain=xtrain.detach().clone(); xval=xval.detach().clone()
    optimizer=torch.optim.AdamW(model.head.parameters(),lr=0.025,weight_decay=0.03)
    criterion=nn.CrossEntropyLoss()
    best_loss=float('inf'); best=None; best_epoch=0
    for epoch in range(epochs):
        order=torch.randperm(len(ytrain))
        model.head.train()
        for indices in order.split(32):
            optimizer.zero_grad(); loss=criterion(model.head(xtrain[indices]),ytrain[indices])
            loss.backward(); optimizer.step()
        model.head.eval()
        with torch.no_grad(): val_loss=float(criterion(model.head(xval),yval))
        if val_loss < best_loss:
            best_loss=val_loss; best=copy.deepcopy(model.head.state_dict()); best_epoch=epoch+1
        if (epoch+1)%20==0: print(f'epoch={epoch+1} val_loss={val_loss:.5f}',flush=True)
    model.head.load_state_dict(best); model.eval()
    # Calibrate with TRAIN views only, including the shape/lighting variation
    # deliberately introduced above. Never use the real-camera audit images.
    calibration_x=torch.cat([xclean,xtrain]) if foreground else xclean
    calibration_y=torch.cat([yclean,ytrain]) if foreground else yclean
    centroids=torch.stack([nn.functional.normalize(calibration_x[calibration_y==c].mean(0),dim=0) for c in range(2)])
    limits=[]
    for c in range(2):
        distances=1-calibration_x[calibration_y==c]@centroids[c]
        limits.append(max(0.08,float(torch.quantile(distances,0.99))*1.3) if foreground else max(0.035,float(torch.quantile(distances,0.99))*1.8))
    limits=torch.tensor(limits)
    with torch.inference_mode():
        probs=model.head(xtest).softmax(1)
        predictions=probs.argmax(1)
        confusion=[[int(((ytest==a)&(predictions==b)).sum()) for b in range(2)] for a in range(2)]
        decisions=[decision(p,x,centroids,limits) for p,x in zip(probs,xtest)]
        coverage=sum(d['class_code']!=3 for d in decisions)/len(decisions)
        accepted=[(d,int(y)) for d,y in zip(decisions,ytest) if d['class_code']!=3]
        accepted_accuracy=sum(d['class_code']==y+1 for d,y in accepted)/len(accepted) if accepted else None
        dummy=torch.zeros(1,3,224,224)
        for _ in range(10): model(dummy)
        timings=[]
        for _ in range(100):
            t=time.perf_counter(); model(dummy); timings.append((time.perf_counter()-t)*1000)
    version=datetime.now(timezone.utc).strftime(('night-' if low_light else '')+('foreground-' if foreground else '')+'mobilenetv3-%Y%m%dT%H%M%SZ')
    report={'model_version':version,'architecture':'frozen MobileNetV3 Small + trained linear head',
            'original_image_count':data['total_original_images'],'split_counts':data['counts'],
            'epochs':epochs,'best_epoch':best_epoch,'augmentation_views_per_train_image':views,
            'test_count':len(ytest),'test_closed_set_accuracy':float((predictions==ytest).float().mean()),
            'test_confusion_matrix_rows_true_cols_pred':confusion,'class_order':CLASSES,
            'test_accepted_coverage':coverage,'test_accepted_accuracy':accepted_accuracy,
            'unknown_threshold':0.85,'unknown_margin':0.25,'embedding_distance_limits':limits.tolist(),
            'feature_transform':'foreground-v1' if foreground else 'full-frame',
            'foreground_config':foreground_config if foreground else None,
            'low_light_augmentation':low_light,
            'original_supplied_image_count':data.get('original_supplied_image_count',data['total_original_images']),
            'real_camera_images_used_for_training':sum(r.get('source') in {'camera_calibration','night_camera'} and r['split']=='train' for r in data['images']),
            'night_camera_images':sum(r.get('source')=='night_camera' for r in data['images']),
            'night_camera_images_used_for_training':sum(r.get('source')=='night_camera' and r['split']=='train' for r in data['images']),
            'cpu_threads':threads,'forward_ms_p50':float(np.percentile(timings,50)),
            'forward_ms_p95':float(np.percentile(timings,95)),
            'training_seconds':time.perf_counter()-started,
            'limitations':['No unknown-class images were used for training; general OOD accuracy is unmeasured.',
                          data['limitations'],'Timing excludes camera, USB, HTTP and RS485 latency.']}
    output.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.state_dict(),'centroids':centroids,'distance_limits':limits,
                'classes':CLASSES,'version':version,'feature_transform':report['feature_transform'],
                'foreground_config':report['foreground_config']},output/'sortnet.pt')
    (output/'training_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); base=Path(__file__).resolve().parents[1]
    p.add_argument('--manifest',type=Path,default=base/'models/dataset_manifest.json')
    p.add_argument('--output',type=Path,default=base/'models')
    p.add_argument('--epochs',type=int,default=160);p.add_argument('--views',type=int,default=6)
    p.add_argument('--full-frame',action='store_true',help='Use the legacy whole-frame pipeline')
    p.add_argument('--low-light',action='store_true',help='Train with exposure/gamma/white-balance/noise variation')
    p.add_argument('--foreground-config',type=Path,help='Foreground preset saved into this candidate checkpoint')
    a=p.parse_args();config=json.loads(a.foreground_config.read_text(encoding='utf-8')) if a.foreground_config else None
    train(a.manifest,a.output,a.epochs,a.views,foreground=not a.full_frame,low_light=a.low_light,foreground_config=config)
