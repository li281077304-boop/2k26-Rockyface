# -*- coding: utf-8 -*-
"""用 MediaPipe FaceLandmarker 提取全部照片的 478 点 landmark，并按头部偏航角分类"""
import os, json, glob
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
PHOTOS = os.path.join(ROOT, "source_photos")
MODELS = os.path.join(ROOT, "models")
OUT = os.path.join(ROOT, "geometry", "photo_landmarks.json")

# MediaPipe C 层无法处理含中文的文件路径，改用内存加载
with open(os.path.join(MODELS, "face_landmarker.task"), "rb") as f:
    MODEL_BYTES = f.read()


def load_image_rgb(path):
    """支持中文路径的图片读取 -> mp.Image(RGB)"""
    buf = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("imdecode failed: " + path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


opts = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_buffer=MODEL_BYTES),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    output_face_blendshapes=False,
)
det = vision.FaceLandmarker.create_from_options(opts)

# 常用 landmark 索引（MediaPipe FaceMesh 478 点规范索引）
IDX = {
    "left_eye_outer":   33,   "left_eye_inner":  133,  "left_pupil": 468,
    "right_eye_outer": 263,   "right_eye_inner": 362,  "right_pupil": 473,
    "nose_bridge":      168,  "nose_tip":          1,
    "left_nostril":     98,   "right_nostril":   327,
    "mouth_left":       61,   "mouth_right":     291,
    "upper_lip":         0,   "lower_lip":        17,
    "chin":             152,  "forehead_top":     10,
    "left_cheek":       234,  "right_cheek":     454,
    "left_ear_top":     127,  "right_ear_top":   356,
    "brow_left_in":     55,   "brow_left_out":   46,
    "brow_right_in":   285,   "brow_right_out": 276,
}

results = {}
for p in sorted(glob.glob(os.path.join(PHOTOS, "*.jpg"))):
    name = os.path.basename(p)
    try:
        img = load_image_rgb(p)
        r = det.detect(img)
        if not r.face_landmarks:
            results[name] = {"status": "no_face"}
            print("%-46s  未检测到人脸" % name)
            continue
        lm = r.face_landmarks[0]
        arr = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float64)
        # 偏航角估计: 鼻尖相对双眼中点的水平偏移 / 双眼间距
        L = arr[IDX["left_eye_outer"]]; R = arr[IDX["right_eye_outer"]]
        eye_mid = (L + R) / 2
        ipd = np.linalg.norm(L[:2] - R[:2])
        nose = arr[IDX["nose_tip"]]
        yaw = (nose[0] - eye_mid[0]) / max(ipd, 1e-9)   # >0 朝右
        # 俯仰: 鼻尖相对双眼中点的垂直偏移 / 眼-下巴距离
        chin = arr[IDX["chin"]]
        pitch = (nose[1] - eye_mid[1]) / max(np.linalg.norm(eye_mid[:2] - chin[:2]), 1e-9)
        rec = {
            "status": "ok", "n_points": len(arr),
            "yaw": round(float(yaw), 4), "pitch": round(float(pitch), 4),
            "landmarks": arr.tolist(),
            "key": {k: arr[v].tolist() for k, v in IDX.items()},
        }
        results[name] = rec
        tag = "正脸" if abs(yaw) < 0.12 else ("右转" if yaw > 0 else "左转")
        tag += "/低头" if pitch > 0.05 else ("/抬头" if pitch < -0.05 else "")
        print("%-46s  %d 点  yaw=%+.3f pitch=%+.3f  %s" % (name, len(arr), yaw, pitch, tag))
    except Exception as e:
        results[name] = {"status": "error", "error": str(e)}
        print("%-46s  ERROR %s" % (name, e))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False)
print("\n已写入", OUT)
