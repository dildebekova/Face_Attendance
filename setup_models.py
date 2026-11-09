"""Explicit model acquisition; review upstream licences before running."""
import argparse
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--accept-research-licenses', action='store_true')
a=p.parse_args()
if not a.accept_research_licenses:
    p.error('Сначала прочитайте MODEL_LICENSES.md. Для исследовательского прототипа добавьте --accept-research-licenses')
root=Path(__file__).parent
models=root/'models';models.mkdir(exist_ok=True)
def download(url,dest):
    if dest.exists(): return
    dest.parent.mkdir(parents=True,exist_ok=True)
    temp=dest.with_suffix(dest.suffix+'.part')
    try:
        print('Загрузка',dest.name)
        urllib.request.urlretrieve(url,temp)
        temp.replace(dest)
    finally: temp.unlink(missing_ok=True)
archive=models/'buffalo_l.zip'
download('https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip',archive)
target=models/'insightface/models/buffalo_l';target.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(archive) as z:
    for name in ['det_10g.onnx','w600k_r50.onnx']:
        candidates=[x for x in z.namelist() if Path(x).name==name]
        if len(candidates)!=1: raise RuntimeError('Неверный архив InsightFace')
        with z.open(candidates[0]) as src,(target/name).open('wb') as dst: shutil.copyfileobj(src,dst)
download('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',models/'face_landmarker.task')
vendor=root/'vendor/Silent-Face-Anti-Spoofing'
if not vendor.exists():
    vendor.parent.mkdir(exist_ok=True)
    subprocess.run(['git','init',str(vendor)],check=True)
    subprocess.run(['git','-C',str(vendor),'remote','add','origin','https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git'],check=True)
    subprocess.run(['git','-C',str(vendor),'fetch','--depth','1','origin','b6d5f04ad78778917853b25c778acef6d5626d15'],check=True)
    subprocess.run(['git','-C',str(vendor),'checkout','--detach','FETCH_HEAD'],check=True)
weights=vendor/'resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth'
if not weights.exists(): raise RuntimeError('В репозитории нет весов MiniFASNetV2. См. upstream README')
(models/'anti_spoof').mkdir(exist_ok=True)
shutil.copyfile(weights,models/'anti_spoof'/weights.name)
commit=subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()
(root/'models/SILENT_FACE_COMMIT.txt').write_text(commit+'\n')
import hashlib,json
manifest={str(f.relative_to(models)):hashlib.sha256(f.read_bytes()).hexdigest() for f in models.rglob('*') if f.suffix in ('.onnx','.task','.pth')}
(models/'sha256.json').write_text(json.dumps(manifest,indent=2))
print('Модели подготовлены. Сверьте источники и лицензии; запускайте docker compose up --build.')
