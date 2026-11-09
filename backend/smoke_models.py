"""Run real models on the bundled scikit-image sample. This is NOT an accuracy test."""
import argparse
import base64
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import cv2
import numpy as np
from skimage import data
from vision import Vision


def run():
    start=time.perf_counter()
    v=Vision();v.load()
    sample=cv2.cvtColor(data.astronaut(),cv2.COLOR_RGB2BGR)
    ok,jpeg=cv2.imencode('.jpg',sample)
    assert ok
    encoded=base64.b64encode(jpeg).decode()
    decoded=v.decode(encoded)
    result=v.diagnose(encoded)
    assert result['faces']==1, result
    assert result['embedding_dimensions']==512, result
    assert result['landmarks_detected'], result
    assert np.isfinite(result['pad_live_score']),result
    faces=v.face.get(decoded)
    a=faces[0].normed_embedding
    b=v.face.get(decoded)[0].normed_embedding
    same=float(np.dot(a,b))
    assert same>0.999
    blank=np.zeros((480,640,3),dtype=np.uint8)
    assert len(v.face.get(blank))==0
    _,j=cv2.imencode('.jpg',blank)
    try: v.inspect(base64.b64encode(j).decode())
    except ValueError: blank_rejected=True
    else: raise AssertionError('Empty image was accepted')
    return {'status':'passed','test_type':'real model execution on static public sample, not physical camera or accuracy test',
            'python':platform.python_version(),'versions':{n:importlib.metadata.version(n) for n in ['insightface','onnxruntime','mediapipe','torch','numpy']},
            'providers':v.providers,'sample_results':result,'same_image_cosine':round(same,6),'blank_rejected':blank_rejected,
            'total_seconds':round(time.perf_counter()-start,2),'physical_camera_tested':False,'face_id_equivalence':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
    result=run();text=json.dumps(result,ensure_ascii=False,indent=2);print(text)
    if args.output: Path(args.output).write_text(text+'\n')
