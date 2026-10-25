from __future__ import annotations
import asyncio, json, re, threading, time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from model import SortNet, preprocess, decision
from foreground import locate_product, draw_search_region

BASE=Path(__file__).resolve().parent
CONFIG=json.loads((BASE/'config.json').read_text(encoding='utf-8'))


class Camera:
    def __init__(self):
        self.frames=deque(maxlen=max(30,int(CONFIG['fps']*CONFIG['buffer_seconds'])))
        self.lock=threading.Lock();self.stop=threading.Event();self.thread=None
        self.error='Camera has not started'

    def start(self):
        self.thread=threading.Thread(target=self._read,daemon=True);self.thread.start()

    def _read(self):
        cap=cv2.VideoCapture(CONFIG['camera_index'],cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release();cap=cv2.VideoCapture(CONFIG['camera_index'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,CONFIG['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CONFIG['height'])
        cap.set(cv2.CAP_PROP_FPS,CONFIG['fps']);cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        try:
            while not self.stop.is_set():
                ok,frame=cap.read()
                if not ok:
                    self.error='Cannot read camera index '+str(CONFIG['camera_index']);time.sleep(0.1);continue
                now=datetime.now(timezone.utc)
                with self.lock: self.frames.append((now,frame))
                self.error=''
        finally: cap.release()

    def frame(self,target=None):
        with self.lock:
            if not self.frames: raise RuntimeError(self.error or 'No camera frame')
            if (datetime.now(timezone.utc)-self.frames[-1][0]).total_seconds()>0.5:
                raise RuntimeError('Camera frame stream is stale')
            timestamp,frame=min(self.frames,key=lambda p:abs((p[0]-target).total_seconds())) if target else self.frames[-1]
            if target and abs((timestamp-target).total_seconds())*1000>CONFIG['maximum_trigger_frame_error_ms']:
                raise RuntimeError('Trigger is outside the camera buffer or clock alignment is incorrect')
            return timestamp,frame.copy()


camera=Camera();model=None;checkpoint=None;pending={};inference_lock=asyncio.Lock()
transform=preprocess(False)


def foreground_config():
    if checkpoint and checkpoint.get('feature_transform')=='foreground-v1':
        return {**checkpoint['foreground_config'],**CONFIG.get('foreground',{})}
    return None


@asynccontextmanager
async def lifespan(app):
    global model,checkpoint
    torch.set_num_threads(CONFIG['torch_threads'])
    path=(BASE/CONFIG['model_path']).resolve()
    if path.exists():
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        model=SortNet();model.load_state_dict(checkpoint['state_dict']);model.eval()
        with torch.inference_mode():
            for _ in range(8):model(torch.zeros(1,3,224,224))
    camera.start()
    yield
    camera.stop.set()
    if camera.thread:camera.thread.join(timeout=2)


app=FastAPI(title='Băng tải — nhận diện',lifespan=lifespan)


class Trigger(BaseModel):
    product_id:str=Field(pattern=r'^[a-zA-Z0-9_-]{1,100}$')
    trigger_at_utc:datetime


@app.get('/health')
def health():
    try: timestamp,_=camera.frame();camera_ok=True
    except RuntimeError:timestamp=None;camera_ok=False
    return {'ready':model is not None and camera_ok,'model_loaded':model is not None,
            'camera_ready':camera_ok,'camera_index':CONFIG['camera_index'],
            'latest_frame_at':timestamp,'error':camera.error,
            'model_version':checkpoint['version'] if checkpoint else None,
            'feature_transform':checkpoint.get('feature_transform','full-frame') if checkpoint else None,
            'foreground_roi':foreground_config().get('roi') if foreground_config() else None}


@app.get('/frame.jpg')
def jpeg():
    try:_,frame=camera.frame()
    except RuntimeError as exc:raise HTTPException(503,str(exc))
    if foreground_config():draw_search_region(frame,foreground_config())
    ok,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85])
    if not ok:raise HTTPException(503,'JPEG encoding failed')
    return Response(encoded.tobytes(),media_type='image/jpeg',headers={'Cache-Control':'no-store'})


@app.get('/stream.mjpg')
async def stream():
    async def generate():
        while True:
            try:
                _,frame=camera.frame()
                if foreground_config():draw_search_region(frame,foreground_config())
                ok,jpg=cv2.imencode('.jpg',frame)
                if ok:yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpg.tobytes()+b'\r\n'
            except RuntimeError:pass
            await asyncio.sleep(0.05)
    return StreamingResponse(generate(),media_type='multipart/x-mixed-replace; boundary=frame')


async def infer(product_id,timestamp,frame):
    async with inference_lock:
        def work():
            start=time.perf_counter()
            rgb=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            settings=foreground_config()
            located=locate_product(rgb,settings) if settings else None
            if settings and located is None:
                result={'class_code':3,'product_type':'unknown','confidence':0.0,'margin':0.0,
                        'embedding_distance':None,'raw_prediction':None,'probabilities':None,
                        'rejection_reasons':['no_product_in_roi']}
            else:
                with torch.inference_mode():
                    embedding=model.embedding(transform(located.image if located else rgb).unsqueeze(0))
                    probabilities=model.head(embedding).softmax(1)[0]
                    result=decision(probabilities,embedding[0],checkpoint['centroids'],checkpoint['distance_limits'],
                                    CONFIG['confidence_threshold'],CONFIG['margin_threshold'])
                    result['probabilities']={name:float(value) for name,value in zip(checkpoint['classes'],probabilities)}
                    result['raw_prediction']=checkpoint['classes'][int(probabilities.argmax())]
            result.update(product_id=product_id,captured_at=timestamp.isoformat(),
                          inference_ms=(time.perf_counter()-start)*1000,model_version=checkpoint['version'],
                          feature_transform=checkpoint.get('feature_transform','full-frame'),
                          foreground_bbox=list(located.bbox) if located else None,
                          foreground_area_fraction=located.area_fraction if located else None,
                          touches_frame_edge=located.touches_frame_edge if located else None)
            target=(BASE/CONFIG['capture_directory']).resolve();target.mkdir(parents=True,exist_ok=True)
            filename=target/(product_id+'.jpg')
            # Python OpenCV imwrite does not support all Unicode paths on Windows.
            ok,encoded=cv2.imencode('.jpg',frame)
            if not ok:raise RuntimeError('Snapshot encoding failed')
            filename.write_bytes(encoded.tobytes());result['image_path']=str(filename)
            if located:
                crop_directory=target/'roi';crop_directory.mkdir(parents=True,exist_ok=True)
                crop_filename=crop_directory/(product_id+'.jpg')
                ok,crop_encoded=cv2.imencode('.jpg',cv2.cvtColor(np.asarray(located.image),cv2.COLOR_RGB2BGR))
                if not ok:raise RuntimeError('Product crop encoding failed')
                crop_filename.write_bytes(crop_encoded.tobytes());result['roi_image_path']=str(crop_filename)
            result['processing_ms']=(time.perf_counter()-start)*1000
            filename.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            return result
        return await asyncio.to_thread(work)


@app.post('/classify')
async def classify(request:Trigger):
    if model is None:raise HTTPException(503,'No trained model. Run train.py first.')
    if request.trigger_at_utc.tzinfo is None:raise HTTPException(422,'trigger_at_utc must include UTC offset')
    if request.product_id not in pending:
        try:timestamp,frame=camera.frame(request.trigger_at_utc.astimezone(timezone.utc))
        except RuntimeError as exc:raise HTTPException(503,str(exc))
        # Capture is fixed before waiting for inference; retries never re-photograph another object.
        pending[request.product_id]=asyncio.create_task(infer(request.product_id,timestamp,frame))
        if len(pending)>2048:
            for key in list(pending):
                if key!=request.product_id and pending[key].done():del pending[key]
                if len(pending)<=1024:break
    try:return await asyncio.shield(pending[request.product_id])
    except Exception as exc:raise HTTPException(503,str(exc))


if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8765,workers=1)
