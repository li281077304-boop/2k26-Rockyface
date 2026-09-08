# -*- coding: utf-8 -*-
"""
在原生 8779 头模上建立 MediaPipe landmark -> 顶点 映射
方法：
  1. 用软件渲染器把原生头正投到 1024x1024
  2. MediaPipe FaceLandmarker 在渲染图上检测 478 点
  3. 每个 landmark 反投影到网格，找最近顶点（z 大者优先，避免后脑）
产物：
  geometry/rocky_landmark_vertex_map.json
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from meshkit import render  # noqa
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import cv2

MODEL = os.path.join(ROOT, "models", "face_landmarker.task")
with open(MODEL, "rb") as f:
    MB = f.read()

vh = np.load(os.path.join(ROOT, "geometry", "native_8779_head_pos.npy"))
fh = np.load(os.path.join(ROOT, "geometry", "head_faces_local.npy"))
SIZE = 1024

# ---- 1) 渲染正面并保存投影参数 ----
yaw0 = 0.0
v = vh.copy()
lo, hi = v.min(0), v.max(0)
span = max(hi[0] - lo[0], hi[1] - lo[1]) * 1.12
cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2


def to_px(pts3d):
    px = ((pts3d[:, 0] - cx) / span + 0.5) * (SIZE - 1)
    py = (1 - (pts3d[:, 1] - cy) / span) * (SIZE - 1)
    return np.stack([px, py], 1)


render(vh, fh, os.path.join(ROOT, "previews", "lm_detect_front.png"), yaw=0, size=SIZE)

# ---- 2) MediaPipe 检测 ----
buf = np.fromfile(os.path.join(ROOT, "previews", "lm_detect_front.png"), dtype=np.uint8)
img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_buffer=MB),
    running_mode=vision.RunningMode.IMAGE, num_faces=1))
r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
assert r.face_landmarks, "渲染图上未检测到人脸"
lm = np.array([[l.x * SIZE, l.y * SIZE] for l in r.face_landmarks[0]])
print("渲染图检测到 %d 个 landmark" % len(lm))

# ---- 3) 反投影：每 landmark 找最近顶点 ----
px_all = to_px(vh)
z = vh[:, 2]
K = 30  # 候选最近顶点数，取其中 z 最大者（同一像素上前面那层）
M = len(lm)
matches = np.full(M, -1, dtype=np.int64)
dist2 = np.full(M, 1e18)

from scipy.spatial import cKDTree
tree = cKDTree(px_all)
dd, ii = tree.query(lm, k=K)
for j in range(M):
    cand = ii[j]
    # 同像素多候选取 z 大（靠前）
    best = cand[np.argmax(z[cand])]
    matches[j] = best
    dist2[j] = dd[j]

good = dist2 < 6.0
print("有效映射 landmark 数: %d / %d (最近距离<6px)" % (good.sum(), M))

# ---- 4) 一级锚点清单 ----
ANCHOR_MP = {
    "left_eye_outer": 33, "left_eye_inner": 133, "right_eye_inner": 362, "right_eye_outer": 263,
    "left_eye_upper": 159, "left_eye_lower": 145, "right_eye_upper": 386, "right_eye_lower": 374,
    "left_brow_in": 55, "left_brow_mid": 52, "left_brow_out": 46,
    "right_brow_in": 285, "right_brow_mid": 282, "right_brow_out": 276,
    "nose_bridge": 168, "nose_tip": 1,
    "left_nostril": 98, "right_nostril": 327,
    "left_mouth_corner": 61, "right_mouth_corner": 291,
    "upper_lip": 0, "lower_lip": 17, "mouth_center": 13,
    "philtrum": 164,
    "chin": 152, "chin_left": 172, "chin_right": 397,
    "left_cheek": 234, "right_cheek": 454,
    "left_jaw": 136, "right_jaw": 361,
    "forehead": 10,
    "left_temple": 234, "right_temple": 454,
}
map_out = {}
for name, mpidx in ANCHOR_MP.items():
    vx = int(matches[mpidx])
    map_out[name] = {"mp_index": int(mpidx), "vertex": vx,
                     "pos": [round(float(x), 4) for x in vh[vx]],
                     "dist_px": float(dist2[mpidx])}

# 顶点组：为每个锚点附带邻域顶点（欧氏半径 ~2.5 单位）
groups = {}
for name, info in map_out.items():
    p = vh[info["vertex"]]
    d = np.linalg.norm(vh - p, axis=1)
    g = np.where(d < 2.5)[0]
    groups[name] = g.tolist()

out = {"n_vertex": len(vh), "size": SIZE, "projection": {"cx": cx, "cy": cy, "span": span},
       "anchors": map_out, "groups": groups,
       "all_matches": {str(i): int(matches[i]) for i in range(M) if good[i]}}

with open(os.path.join(ROOT, "geometry", "rocky_landmark_vertex_map.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
print("已写入 geometry/rocky_landmark_vertex_map.json")
print("关键锚点：")
for name in ["nose_tip", "nose_bridge", "chin", "left_eye_outer", "right_eye_outer",
             "left_mouth_corner", "right_mouth_corner", "upper_lip", "lower_lip", "forehead"]:
    print("  %-20s vertex=%6d  pos=%s  dist=%.2fpx"
          % (name, map_out[name]["vertex"], map_out[name]["pos"], map_out[name]["dist_px"]))
