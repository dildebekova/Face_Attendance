"""Real inference only. No fallback identities or simulated liveness."""
import base64
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(os.getenv("MODEL_ROOT", "/models"))

class Vision:
    def __init__(self):
        self.ready = False
        self.error = "Модели ещё не загружены"

    def load(self):
        if self.ready:
            return
        try:
            self.liveness_enabled = os.getenv("ENABLE_LIVENESS", "0").lower() in ("1", "true", "yes", "on")
            self.liveness_enforcement = self.liveness_enabled
            required = [ROOT / "insightface/models/buffalo_l/det_10g.onnx",
                        ROOT / "insightface/models/buffalo_l/w600k_r50.onnx",
                        ROOT / "face_landmarker.task"]
            if self.liveness_enabled:
                required.append(ROOT / "anti_spoof/2.7_80x80_MiniFASNetV2.pth")
            missing = [str(x) for x in required if not x.is_file()]
            if missing:
                raise RuntimeError("Отсутствуют модели: " + ", ".join(missing))
            from insightface.app import FaceAnalysis
            import mediapipe as mp
            import onnxruntime as ort
            self.providers = ort.get_available_providers()
            self.face = FaceAnalysis(name="buffalo_l", root=str(ROOT / "insightface"),
                                     allowed_modules=["detection", "recognition"],
                                     providers=["CPUExecutionProvider"])
            self.face.prepare(ctx_id=-1, det_size=(640, 640))
            self.mp = mp
            self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
                mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT / "face_landmarker.task")),
                    num_faces=1))
            vendor = Path(os.getenv("SILENT_FACE_ROOT", "/vendor/Silent-Face-Anti-Spoofing"))
            if not (vendor / "src/anti_spoof_predict.py").is_file():
                raise RuntimeError("Нет исходников Silent-Face-Anti-Spoofing. Запустите setup_models.py")
            sys.path.insert(0, str(vendor))
            from src.model_lib.MiniFASNet import MiniFASNetV2
            import torch
            from src.generate_patches import CropImage
            self.torch = torch
            self.anti = MiniFASNetV2(conv6_kernel=(5, 5))
            state = torch.load(str(ROOT / "anti_spoof/2.7_80x80_MiniFASNetV2.pth"), map_location="cpu", weights_only=True)
            state = {k.removeprefix("module."): v for k,v in state.items()}
            self.anti.load_state_dict(state)
            self.anti.eval()
            self.cropper = CropImage()
            self.ready = True
            self.error = ""
        except Exception as exc:
            self.error = str(exc)
            raise RuntimeError(self.error) from exc

    @staticmethod
    def decode(encoded):
        raw = base64.b64decode(encoded.split(",")[-1], validate=True)
        if len(raw) > 750_000:
            raise ValueError("Кадр слишком большой")
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None or max(img.shape[:2]) > 1280:
            raise ValueError("Неверный размер кадра")
        return img

    def _inspect(self, encoded, enforce_quality=True):
        img = self.decode(encoded)
        faces = self.face.get(img)
        if len(faces) != 1:
            raise ValueError("В кадре должен быть ровно один человек")
        face = faces[0]
        x1,y1,x2,y2 = face.bbox.astype(int)
        x1,y1=max(0,x1),max(0,y1)
        x2,y2=min(img.shape[1],x2),min(img.shape[0],y2)
        if min(x2-x1,y2-y1) < 110:
            raise ValueError("Подойдите ближе к камере")
        gray = cv2.cvtColor(img[y1:y2,x1:x2], cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray,cv2.CV_64F).var())
        sharpness_threshold = float(os.getenv("SHARPNESS_THRESHOLD", "20"))
        if enforce_quality and sharpness < sharpness_threshold:
            raise ValueError("Изображение размыто. Не двигайтесь слишком быстро")
        if not 45 < gray.mean() < 220:
            raise ValueError("Нужно ровное освещение лица")
        pad_score = 1.0
        if getattr(self, "liveness_enabled", False):
            prediction = self.pad(img, [x1,y1,x2-x1,y2-y1])
            pad_score = float(prediction[1])
            if getattr(self, "liveness_enforcement", False) and enforce_quality and (int(np.argmax(prediction)) != 1 or pad_score < float(os.getenv("PAD_THRESHOLD", "0.85"))):
                raise ValueError("Проверка живого лица не пройдена. Повторите попытку")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            raise ValueError("Не удалось проверить положение головы")
        pts = result.face_landmarks[0]
        # Ratio in unmirrored camera coordinates; UI mirrors only the preview.
        eye_mid = (pts[33].x + pts[263].x) / 2
        yaw = (pts[1].x - eye_mid) / max(abs(pts[263].x-pts[33].x), 0.001)
        v = face.normed_embedding.astype(float)
        if not np.isfinite(v).all():
            raise ValueError("Неверный шаблон лица")
        return v, float(yaw), sharpness, pad_score

    def inspect(self, encoded):
        v, yaw, _, _ = self._inspect(encoded, enforce_quality=True)
        return v, yaw

    def sequence(self, frames, direction):
        self.load()
        observations = []
        errors = []
        for encoded in frames:
            try:
                observations.append(self._inspect(encoded, enforce_quality=False))
            except ValueError as exc:
                errors.append(str(exc))
        min_frames = max(4, int(os.getenv("MIN_VALID_FRAMES", "4")))
        if len(observations) < min_frames:
            raise ValueError(errors[-1] if errors else "Не удалось проверить кадры")
        sharpness_threshold = float(os.getenv("SHARPNESS_THRESHOLD", "20"))
        sharp = [x for x in observations if x[2] >= sharpness_threshold]
        if len(sharp) < min_frames:
            raise ValueError("Кадры слишком размыты. Замедлите поворот головы")
        if getattr(self, "liveness_enforcement", False):
            pad_threshold = float(os.getenv("PAD_THRESHOLD", "0.85"))
            live = [x for x in sharp if x[3] >= pad_threshold]
            if len(live) < min_frames:
                raise ValueError("Проверка живого лица не пройдена. Повторите попытку")
        else:
            live = sharp
        vectors = [x[0] for x in live]
        yaws = [x[1] for x in live]
        baseline = float(np.mean(yaws[:2]))
        if abs(baseline) > 0.18:
            raise ValueError("В начале проверки смотрите прямо в камеру")
        shift = float(np.mean(yaws[-2:])) - baseline
        if direction * shift < 0.12:
            raise ValueError("Поверните голову по стрелке после отсчёта")
        if any(float(np.dot(vectors[0],v)) < 0.45 for v in vectors[1:]):
            raise ValueError("Лицо изменилось во время проверки")
        v = np.mean(vectors, axis=0)
        return v / np.linalg.norm(v)

    def pad(self, img, bbox):
        patch = self.cropper.crop(org_img=img, bbox=bbox,
                                  scale=2.7, out_w=80, out_h=80, crop=True)
        # Match upstream numpy ToTensor: BGR CHW float32 in 0..255, NOT /255.
        tensor = self.torch.from_numpy(patch.transpose(2,0,1).copy()).float().unsqueeze(0)
        with self.torch.inference_mode():
            prediction = self.torch.softmax(self.anti(tensor), dim=1).numpy()[0]
        return prediction

    def diagnose(self, encoded):
        self.load()
        start = time.perf_counter()
        img = self.decode(encoded)
        faces = self.face.get(img)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        metrics = {"faces": len(faces), "landmarks_detected": bool(result.face_landmarks),
                   "width": img.shape[1], "height": img.shape[0], "attendance_written": False}
        if len(faces) == 1:
            f = faces[0]
            x1,y1,x2,y2 = f.bbox.astype(int)
            x1,y1=max(0,x1),max(0,y1); x2,y2=min(img.shape[1],x2),min(img.shape[0],y2)
            gray=cv2.cvtColor(img[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY)
            metrics.update(embedding_dimensions=len(f.normed_embedding),
                           brightness=round(float(gray.mean()),1),
                           sharpness=round(float(cv2.Laplacian(gray,cv2.CV_64F).var()),1))
            if self.liveness_enforcement:
                prediction = self.pad(img,[x1,y1,x2-x1,y2-y1])
                metrics.update(pad_live_score=round(float(prediction[1]),4),
                               pad_pass=bool(np.argmax(prediction)==1 and prediction[1]>=float(os.getenv("PAD_THRESHOLD","0.85"))))
                metrics["detail"] = ("Проверка PAD на этом кадре пройдена." if metrics["pad_pass"] else "Проверка PAD на этом кадре не пройдена.") + " Для отметки нужна отдельная проверка с поворотом головы."
            else:
                metrics["detail"] = "Проверка живого лица отключена для настройки распознавания."
        else:
            metrics["detail"] = "В кадре должен быть ровно один человек. Подойдите ближе и проверьте свет."
        metrics["elapsed_ms"] = round((time.perf_counter()-start)*1000)
        return metrics
