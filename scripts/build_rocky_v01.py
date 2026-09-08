# -*- coding: utf-8 -*-
"""
Rocky v0.3a —— 头模几何形变第一版

方法：
  1. 从原生 8779 头模（已切出）的中央 profile 自动定位解剖点
     （鼻尖/下巴/眉骨/嘴角/下颌角/颧骨）
  2. 以这些点为 anchor，按"成年骨架上的儿童脸"比例设定目标位置
  3. RBF（薄板样条）求连续形变场，禁止整体 X/Y/Z 缩放
  4. 用软权重掩码保证颈部/后脑/头顶锁定
  5. 输出 rocky_target_v02.obj + 四视角预览
"""
import os, sys
import numpy as np
from scipy.interpolate import RBFInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from meshkit import read_obj, write_obj, render, contact_sheet  # noqa

vh = np.load(os.path.join(ROOT, "geometry", "native_8779_head_pos.npy"))
fh = np.load(os.path.join(ROOT, "geometry", "head_faces_local.npy"))
print("头模:", vh.shape, fh.shape)

cx = float(np.median(vh[:, 0]))
Zmax = vh[:, 2].max()

# ---------- 1) 中央 profile ----------
prof = vh[(np.abs(vh[:, 0] - cx) < 1.3) & (vh[:, 2] > 1.0)]
prof = prof[np.argsort(prof[:, 1])]
nose = prof[np.argmax(prof[:, 2])]
front = prof[prof[:, 2] > 0.55 * Zmax]
chin = front[np.argmin(front[:, 1])]
face_h = float(nose[1] - chin[1])
face_w = float(vh[:, 0].max() - vh[:, 0].min())
print("鼻尖 Y=%.2f Z=%.2f | 下巴 Y=%.2f | 脸高=%.2f 脸宽=%.2f"
      % (nose[1], nose[2], chin[1], face_h, face_w))

# 眉骨：鼻尖上方 0.40~0.75 脸高内 Z 的局部峰
band = prof[(prof[:, 1] > nose[1] + 0.40 * face_h) & (prof[:, 1] < nose[1] + 0.80 * face_h)]
brow = band[np.argmax(band[:, 2])] if len(band) else nose + np.array([0, 0.45 * face_h, 0])
nose_h = float(nose[2] - brow[2])       # 鼻尖比眉骨前凸多少（正值）
if nose_h < 0.5:
    nose_h = 1.0
print("眉骨 Y=%.2f Z=%.2f | 鼻凸=%.2f" % (brow[1], brow[2], nose_h))

# ---------- 2) 左右 anchor ----------
def side_anchor(y_lo, y_hi, z_min=0.0):
    m = (vh[:, 1] > y_lo) & (vh[:, 1] < y_hi) & (vh[:, 2] > z_min)
    p = vh[m]
    if not len(p):
        return None
    return p[np.argmax(np.abs(p[:, 0] - cx))]


jaw_y0, jaw_y1 = chin[1] + 0.08 * face_h, chin[1] + 0.38 * face_h
jawL, jawR = side_anchor(jaw_y0, jaw_y1, 2.0), side_anchor(jaw_y0, jaw_y1, 2.0)
mouth_y = chin[1] + 0.52 * face_h
mL = side_anchor(mouth_y - 2, mouth_y + 2, 6.0)
mR = side_anchor(mouth_y - 2, mouth_y + 2, 6.0)
cheekL = side_anchor(nose[1] - 0.2 * face_h, nose[1] + 0.2 * face_h, 4.0)
cheekR = side_anchor(nose[1] - 0.2 * face_h, nose[1] + 0.2 * face_h, 4.0)
browL = side_anchor(brow[1] - 2, brow[1] + 2, 4.0)
browR = side_anchor(brow[1] - 2, brow[1] + 2, 4.0)

# ---------- 3) anchor 目标 ----------
S = lambda f: f * face_h          # 脸高比例
N = lambda f: f * max(nose_h, 1.0)
A, T = [], []                     # anchor 原位置 / 目标位置


def add(p, d):
    A.append(p); T.append(p + np.array(d))


add(chin,  [0, S(0.30), -N(0.05)])                    # 下巴上提、略收
if jawL is not None: add(jawL, [-S(0.05), S(0.24), 0])
if jawR is not None: add(jawR, [S(0.05), S(0.24), 0])
if mL is not None:  add(mL,   [0, S(0.14), 0])
if mR is not None:  add(mR,   [0, S(0.14), 0])
add(nose,  [0, N(0.06), -N(0.16)])                    # 鼻尖上提、凸出度下降
add(np.array([cx, (nose[1]+brow[1])/2, nose[2]-N(0.3)]), [0, 0, -N(0.10)])   # 鼻梁压低
if browL is not None: add(browL, [0, 0, -N(0.07)])
if browR is not None: add(browR, [0, 0, -N(0.07)])
if cheekL is not None: add(cheekL, [S(0.015), 0, N(0.04)])   # 颊部略饱满
if cheekR is not None: add(cheekR, [-S(0.015), 0, N(0.04)])

A = np.array(A); T = np.array(T)
print("移动 anchor %d 个" % len(A))

# ---------- 4) 锁定锚点（颈部/后脑/头顶不动）----------
lock_mask = (vh[:, 1] < chin[1] - 0.05 * face_h) | (vh[:, 2] < -2.5) | (vh[:, 1] > brow[1] + 0.95 * face_h)
li = np.random.default_rng(0).choice(np.where(lock_mask)[0], size=min(1500, lock_mask.sum()), replace=False)
A = np.vstack([A, vh[li]])
T = np.vstack([T, vh[li]])
print("锁定锚点 %d 个" % len(li))

# ---------- 5) RBF 形变 ----------
print("求解 RBF ...")
rbf = RBFInterpolator(A, T, kernel="thin_plate_spline", smoothing=0.5,
                      neighbors=min(120, len(A)))
disp = rbf(vh) - vh

# ---------- 6) 软权重掩码 ----------
def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


wy = sstep(chin[1] - 0.10 * face_h, chin[1] + 0.05 * face_h, vh[:, 1]) * \
     (1 - sstep(brow[1] + 0.85 * face_h, brow[1] + 1.05 * face_h, vh[:, 1]))
wz = sstep(-2.0, 1.5, vh[:, 2])
wx = 1 - sstep(0.86 * face_w / 2, 1.0 * face_w / 2, np.abs(vh[:, 0] - cx))
w = wy * wz * wx

out = vh + disp * w[:, None]
print("最大位移 %.2f  平均位移 %.2f" % (np.linalg.norm(disp * w[:, None], axis=1).max(),
                                       np.linalg.norm(disp * w[:, None], axis=1).mean()))

# ---------- 7) 输出 ----------
write_obj(os.path.join(ROOT, "geometry", "rocky_target_v02_head.obj"), out, fh)

# 写回全身：头部顶点替换，其余不动
full = np.load(os.path.join(ROOT, "geometry", "head_vertex_index.npy"))
body = read_obj(os.path.join(ROOT, "geometry", "native_8779.obj"))[0]
body[full] = out
write_obj(os.path.join(ROOT, "geometry", "rocky_target_v02.obj"), body,
          read_obj(os.path.join(ROOT, "geometry", "native_8779.obj"))[1])

views = []
for tag, yaw in [("front", 0), ("left45", -45), ("right45", 45), ("side_left", -90)]:
    p = os.path.join(ROOT, "previews", "rocky02_%s.png" % tag)
    render(out, fh, p, yaw=yaw)
    views.append(p)
contact_sheet(views, os.path.join(ROOT, "previews", "rocky02_sheet.png"))

cmp_paths = [os.path.join(ROOT, "previews", "head_front.png"),
             os.path.join(ROOT, "previews", "rocky02_front.png"),
             os.path.join(ROOT, "previews", "head_side_left.png"),
             os.path.join(ROOT, "previews", "rocky02_side_left.png")]
contact_sheet(cmp_paths, os.path.join(ROOT, "previews", "compare_v02.png"))
print("完成")
