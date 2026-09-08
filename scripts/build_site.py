# -*- coding: utf-8 -*-
"""生成目录式标注器数据：
annotator/data/{manifest.json, landmarks.json, vertices.bin, mirror.bin, firsttri.bin,
                <view>.png, <view>_vid.bin, <view>_vpix.bin}
"""
import os, sys, json, io
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from meshkit import render_pick  # noqa
from PIL import Image

SIZE = 800
vh = np.load(os.path.join(ROOT, "geometry", "native_8779_head_pos.npy"))
fh = np.load(os.path.join(ROOT, "geometry", "head_faces_local.npy"))
N = len(vh)

from scipy.spatial import cKDTree
tree = cKDTree(vh)
ref = vh.copy(); ref[:, 0] = -ref[:, 0]
_, mirror_idx = tree.query(ref)

firsttri = np.full(N, -1, np.int32)
for t in range(len(fh)):
    for k in fh[t]:
        if firsttri[k] < 0:
            firsttri[k] = t

VIEWS = [("front", 0, 0), ("left45", -45, 0), ("right45", 45, 0),
         ("side_left", -90, 0), ("side_right", 90, 0)]

OUT = os.path.join(ROOT, "annotator", "data")
os.makedirs(OUT, exist_ok=True)

manifest = {"schema": 3, "size": SIZE, "nvert": N, "views": []}
for name, yaw, pitch in VIEWS:
    rp = render_pick(vh, fh, yaw=yaw, pitch=pitch, size=SIZE)
    img = rp["img"]
    tri = rp["tri"]; w = rp["w"]
    vmap = np.full((SIZE, SIZE), 65535, np.uint16)
    vis = tri >= 0
    vmap[vis] = fh[tri[vis], np.argmax(w[vis], 1)]
    # 顶点像素表
    import math
    rad = math.radians(yaw)
    c, s = math.cos(rad), math.sin(rad)
    vv = vh @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]).T
    lo, hi = vv.min(0), vv.max(0)
    span = max(hi[0] - lo[0], hi[1] - lo[1]) * 1.12
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    px = ((vv[:, 0] - cx) / span + 0.5) * (SIZE - 1)
    py = (1 - (vv[:, 1] - cy) / span) * (SIZE - 1)
    vpix = np.stack([px, py], 1).astype("<f4")

    Image.fromarray(img).save(os.path.join(OUT, name + ".png"))
    vmap.astype("<u2").tofile(os.path.join(OUT, name + "_vid.bin"))
    vpix.astype("<f4").tofile(os.path.join(OUT, name + "_vpix.bin"))
    manifest["views"].append({"name": name, "img": name + ".png",
                              "vid": name + "_vid.bin", "vpix": name + "_vpix.bin"})
    print("view %-10s ok" % name)

vh.astype("<f4").tofile(os.path.join(OUT, "vertices.bin"))
mirror_idx.astype("<u2").tofile(os.path.join(OUT, "mirror.bin"))
firsttri.astype("<i4").tofile(os.path.join(OUT, "firsttri.bin"))

# ------- landmark 定义 -------
L = [
    ["left_eye_outer", "左外眼角", "眼睛最外侧角。", "right_eye_outer", "front"],
    ["left_eye_inner", "左内眼角", "眼睛靠鼻侧角。", "right_eye_inner", "front"],
    ["left_brow_in", "左眉头", "眉毛靠鼻起点。", "right_brow_in", "front"],
    ["left_brow_arch", "左眉峰", "眉毛最高弯点。", "right_brow_arch", "front"],
    ["left_nostril", "左鼻翼", "鼻孔外侧边缘。", "right_nostril", "front"],
    ["left_mouth_corner", "左嘴角", "嘴缝最左端。", "right_mouth_corner", "front"],
    ["left_jaw_angle", "左下颌角", "下颌角(耳前下方)。", "right_jaw_angle", "front"],
    ["left_cheekbone", "左颧骨", "颧骨外缘(眼下外侧)。", "right_cheekbone", "front"],
    ["nose_root", "鼻根", "两眼间眉心下凹陷。", None, "front"],
    ["nose_tip", "鼻尖", "正面标；可到侧视图补标定深度。", None, "front"],
    ["upper_lip", "上唇中心", "人中下缘中点。", None, "front"],
    ["lower_lip", "下唇中心", "下唇最下缘中点。", None, "front"],
    ["chin", "下巴尖", "正面标；可到侧视图补标定前突。", None, "front"],
    ["left_ear_top", "左耳顶", "切到 side_left 视图标左耳顶。", None, "side_left"],
    ["left_ear_bottom", "左耳底", "切到 side_left 视图标左耳垂底。", None, "side_left"],
]
R = [
    ["right_eye_outer", "右外眼角", "自动镜像；不准可点击修正。", None, "front"],
    ["right_eye_inner", "右内眼角", "自动镜像。", None, "front"],
    ["right_brow_in", "右眉头", "自动镜像。", None, "front"],
    ["right_brow_arch", "右眉峰", "自动镜像。", None, "front"],
    ["right_nostril", "右鼻翼", "自动镜像。", None, "front"],
    ["right_mouth_corner", "右嘴角", "自动镜像。", None, "front"],
    ["right_jaw_angle", "右下颌角", "自动镜像。", None, "front"],
    ["right_cheekbone", "右颧骨", "自动镜像。", None, "front"],
    ["right_ear_top", "右耳顶", "切到 side_right 视图标。", None, "side_right"],
    ["right_ear_bottom", "右耳底", "切到 side_right 视图标。", None, "side_right"],
]
LAND = L + R
mirror_target = {item[0]: item[3] for item in L if item[3]}
# 自动镜像源 → 目标
MSRC = {}
for k, zh, hint, mk, view in L:
    if mk:
        MSRC[mk] = k

land_out = {"list": LAND, "mirror_target": mirror_target, "mirror_source": MSRC}
with open(os.path.join(OUT, "landmarks.json"), "w", encoding="utf-8") as f:
    json.dump(land_out, f, ensure_ascii=False)
with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False)
print("生成完成 ->", OUT)
