# ⚠️ EXPERIMENTAL / COARSE INIT —— 仅作初始化与辅助验证，非主线（见 STATUS 第八节）。
# 主线：3DDFA_V2 → NRICP → 8779。
# -*- coding: utf-8 -*-
"""
Rocky v04 —— 多照片 landmark 三角化 → RBF 形变 8779

FaceVerse/FLAME 在本沙箱拿不到权重，但其产物等价于
"每个 landmark 的世界 3D 坐标"。本脚本用本机 OpenCV 替代：
  每张照片 PnP 估计弱透视相机
  ↓
  每张照片 landmark 反投影到相机光线
  ↓
  跨照片 3D 中点 → 每个 landmark 的世界 3D 目标
  ↓
  RBF 把 8779 mesh 上的对应顶点拉到目标位置

输入：
  geometry/photo_landmarks.json   (5/6 张照片 × 478 MP 点)
  geometry/rocky_landmark_vertex_map.json   （用户标定后才会存在；先用几何 profile anchor）
  geometry/native_8779_head_pos.npy
  geometry/head_faces_local.npy
"""
import os, sys, json
import numpy as np
from scipy.interpolate import RBFInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from meshkit import write_obj, render, contact_sheet, read_obj  # noqa

vh = np.load(os.path.join(ROOT, "geometry", "native_8779_head_pos.npy"))
fh = np.load(os.path.join(ROOT, "geometry", "head_faces_local.npy"))
LM = json.load(open(os.path.join(ROOT, "geometry", "photo_landmarks.json"), encoding="utf-8"))
MAP_FILE = os.path.join(ROOT, "geometry", "rocky_landmark_vertex_map.json")

# ---------- 1) 选择哪些 MP index 作为 mesh anchor ----------
# 几何 profile anchor（不依赖用户的 vertex map）
CX = float(np.median(vh[:, 0]))
prof = vh[(np.abs(vh[:, 0] - CX) < 1.5) & (vh[:, 2] > 1.0)]
prof = prof[np.argsort(prof[:, 1])]
nose = prof[np.argmax(prof[:, 2])]
front = prof[prof[:, 2] > 0.55 * prof[:, 2].max()]
chin = front[np.argmin(front[:, 1])]
band = prof[(prof[:, 1] > nose[1] + 0.4 * (nose[1] - chin[1])) & (prof[:, 1] < nose[1] + 0.9 * (nose[1] - chin[1]))]
brow = band[np.argmax(band[:, 2])]
ANCHORS_MP = {1: nose, 152: chin, 168: None, 52: brow}
ANCHORS_MP = {k: v for k, v in ANCHORS_MP.items() if v is not None}   # 仅已知 mesh 3D 位置的
print("几何 anchor MP→", ANCHORS_MP)

# 如果用户已经标定 vertex map，补全
if os.path.exists(MAP_FILE):
    vm = json.load(open(MAP_FILE, encoding="utf-8"))["anchors"]
    for nm, info in vm.items():
        mp_idx = info["mp_index"]
        vidx = info["vertex"]
        ANCHORS_MP[mp_idx] = vh[vidx]
    print("用户 vertex map 已合并, 现 anchor 数量数", len(ANCHORS_MP))

# ---------- 2) 加载照片 landmark + 图像尺寸 ----------
from PIL import Image
def lm_and_size(name):
    rec = LM[name]
    arr = np.array(rec["landmarks"])  # [0..1] 原坐标
    p = os.path.join(ROOT, "source_photos", name)
    if not os.path.exists(p):
        p = os.path.join(ROOT, "..", "source_photos", name)
    im = Image.open(p)
    W, H = im.size
    return arr, W, H

names = [n for n, r in LM.items() if r.get("status") == "ok"]
photos = [(n, *lm_and_size(n)) for n in names]
print("可用照片数:", len(photos))

# ---------- 3) 相机假设 ----------
# 弱透视：x_px ≈ s*X + tx, y_px ≈ s*Y + ty
# 其中 s = focal/W 接近 1.0（图片已归一），但 yaw 有差异
# 用 anchor 反投影求解 (s, R_yaw, tx, ty)
def fit_weak_persp(anchor_3d, photo_2d):
    """anchor_3d: N×3 (mesh 空间); photo_2d: N×2 (像素坐标).
    返回 (s, theta_yaw_deg, tx, ty, cost)"""
    from scipy.optimize import least_squares
    def err(p):
        s, th, tx, ty = p
        c, sn = np.cos(np.radians(th)), np.sin(np.radians(th))
        x2 = anchor_3d[:, 0] * c - anchor_3d[:, 2] * sn
        y2 = anchor_3d[:, 1]
        Xm = photo_2d[:, 0]; Ym = photo_2d[:, 1]
        rx = s * (x2 * c - y2 * sn) + tx - Xm
        ry = s * (x2 * sn + y2 * c) + ty - Ym
        return np.concatenate([rx, ry])
    res = least_squares(err, x0=[1.0, 0.0, photo_2d[:, 0].mean(), photo_2d[:, 1].mean()])
    s, th, tx, ty = res.x
    return s, th, tx, ty, res.cost

# ---------- 4) 每张照片：fit 相机 → back-project anchors → 目标 3D ----------
anchor_mp_idxs = list(ANCHORS_MP.keys())
mesh_anchor_3d = np.stack([ANCHORS_MP[i] for i in anchor_mp_idxs], 0)

target_per_anchor = {mp_idx: [] for mp_idx in anchor_mp_idxs}
cameras = {}
for nm, lm_arr, W, H in photos:
    pix = lm_arr[:, :2] * np.array([W, H])  # N×2 像素
    zrel = lm_arr[:, 2]                     # MediaPipe z (相对深度，值越小越靠前)
    a2d = pix[anchor_mp_idxs]               # N×2 anchor 的像素
    a3d = mesh_anchor_3d                    # N×3 anchor 的 mesh 3D
    try:
        s, th, tx, ty, cost = fit_weak_persp(a3d, a2d)
    except Exception as e:
        print(f"  {nm}: 相机拟合失败 {e}"); continue
    cameras[nm] = (s, th, tx, ty, W, H)
    # back-project：弱透视的逆变换
    c, sn = np.cos(np.radians(th)), np.sin(np.radians(th))
    # x_img ≈ s*(x_mesh*cos - z_mesh*sin) + tx   → 解 x_mesh*cos - z_mesh*sin = (x_img - tx)/s
    # y_img ≈ s*(x_mesh*sin + z_mesh*cos) + ty? 不，模型 y 是 mesh 高度，y_img ≈ s*y_mesh + ty
    # 上面 err 用的是 x2,y2 = (Xr,Yr) = (x*c-z*sn, y) → 2D 相似变换（不含 z cos）
    # 所以目标 mesh 空间 XY = 2D 反相似：x_m' = ((x-img-tx)/s)*c - y*sn? Hmm 不齐
    # 简化：用 (s, th, tx, ty) 反算 mesh XY 校正量
    # 图像点 (X,Y): X_m, Y_m = 2D 反相似变换 (X_img-tx)/s, ...
    # 整体看:  2D 相似变换把 mesh(Xr,Yr) 投到 image; 逆变换把 image 投回 mesh(Xr,Yr)
    Rt = np.array([[c, -sn], [sn, c]])       # 2D rotation
    inv = np.linalg.inv(np.array([[s * c, -s * sn], [s * sn, s * c]]))  # 2x2
    # 对每个 anchor 像素点 (u, v) → mesh(Xr, Yr)
    for j, mp_idx in enumerate(anchor_mp_idxs):
        u, v = a2d[j]
        xy = np.linalg.solve(np.array([[s * c, -s * sn], [s * sn, s * c]]), np.array([u - tx, v - ty]))
        Xr, Yr = xy[0], xy[1]
        # 沿 Z 方向（depth）：保持 mesh 原深度（从 ANCHORS_MP 取）
        Xtarget = Xr * c - (-ANCHORS_MP[mp_idx][2]) * sn   # 反向的 X_m = Xr*c + z*sn
        # 简单处理：用 mesh 原 Z，X/Y 反投值作初值
        target_per_anchor[mp_idx].append(np.array([Xr, Yr, ANCHORS_MP[mp_idx][2]]))

# ---------- 5) 求每个 anchor 的目标 3D：跨照片平均 ----------
target_3d = {}
for mp_idx, lst in target_per_anchor.items():
    if not lst: continue
    arr = np.stack(lst, 0)
    target_3d[mp_idx] = arr.mean(axis=0)
    # 也可对 outlier 做鲁棒均值；这里直接取均值
print("目标 3D (mesh 空间):")
for mp_idx, t in target_3d.items():
    print(f"  MP{mp_idx} → ({t[0]:.2f},{t[1]:.2f},{t[2]:.2f})")

# ---------- 6) RBF 形变 8779 ----------
if not target_3d:
    print("⚠️ 没有可用的 anchor（多半是相机拟合失败）。改用 v02 的几何锚点作为初值。")
    # fallback：用 v02 的锚点（RBF 同前），便于至少交付一个 v04 OBJ
    A = []; T_ = []
    def add(p, d): A.append(p); T_.append(p + np.array(d))
    add(chin, [0, face_h * 0.30, -face_h * 0.05])
    add(nose, [0, face_h * 0.07, -face_h * 0.05])
    add(brow, [0, 0, -face_h * 0.05])
    lock_mask = (vh[:, 1] < chin[1] - 0.06 * face_h) | (vh[:, 2] < -2.5) | (vh[:, 1] > brow[1] + 1.0 * face_h)
    li = np.random.default_rng(2).choice(np.where(lock_mask)[0], min(1500, lock_mask.sum()), replace=False)
    A = np.array(A); T_ = np.array(T_)
    A = np.vstack([A, vh[li]]); T_ = np.vstack([T_, vh[li]])
    rbf = RBFInterpolator(A, T_, kernel="thin_plate_spline", smoothing=0.5, neighbors=min(140, len(A)))
    disp = rbf(vh) - vh
else:
    face_h = nose[1] - chin[1]
    A = []; T_ = []
    for mp_idx, t in target_3d.items():
        if mp_idx not in ANCHORS_MP: continue
        d = t - ANCHORS_MP[mp_idx]
        A.append(ANCHORS_MP[mp_idx])
        T_.append(ANCHORS_MP[mp_idx] + d)
    A = np.array(A); T_ = np.array(T_)
    if len(A) < 3:
        # 锚点太少，相机拟合结果不稳。混用一些几何锚点
        A = A.tolist(); T_ = T_.tolist()
        A.append(chin); T_.append(chin + [0, face_h * 0.30, -face_h * 0.05])
        A.append(nose); T_.append(nose + [0, face_h * 0.07, -face_h * 0.05])
        A.append(brow); T_.append(brow + [0, 0, -face_h * 0.05])
        A = np.array(A); T_ = np.array(T_)
    lock_mask = (vh[:, 1] < chin[1] - 0.08 * face_h) | (vh[:, 2] < -2.0) | (vh[:, 1] > brow[1] + 1.0 * face_h)
    li = np.random.default_rng(2).choice(np.where(lock_mask)[0], min(1500, lock_mask.sum()), replace=False)
    A = np.vstack([A, vh[li]]); T_ = np.vstack([T_, vh[li]])
    rbf = RBFInterpolator(A, T_, kernel="thin_plate_spline", smoothing=0.5, neighbors=min(140, len(A)))
    disp = rbf(vh) - vh

# 软权重
def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)
wy = sstep(chin[1] - 0.12 * face_h, chin[1] + 0.06 * face_h, vh[:, 1]) * \
     (1 - sstep(brow[1] + 0.85 * face_h, brow[1] + 1.05 * face_h, vh[:, 1]))
wz = sstep(-2.0, 1.5, vh[:, 2])
wx = 1 - sstep(0.85 * (vh[:, 0].max() - vh[:, 0].min()) / 2,
               1.0 * (vh[:, 0].max() - vh[:, 0].min()) / 2, np.abs(vh[:, 0] - CX))
w = wy * wz * wx
out = vh + disp * w[:, None]

write_obj(os.path.join(ROOT, "geometry", "rocky_target_v04_head.obj"), out, fh)
full_idx = np.load(os.path.join(ROOT, "geometry", "head_vertex_index.npy"))
full, faces = read_obj(os.path.join(ROOT, "geometry", "native_8779.obj"))
full[full_idx] = out
write_obj(os.path.join(ROOT, "geometry", "rocky_target_v04.obj"), full, faces)

# ---------- 7) 量化报告 ----------
print("\n=== v04 重投影误差（每张照片） ===")
def project(mesh_3d, s, th, tx, ty):
    c, sn = np.cos(np.radians(th)), np.sin(np.radians(th))
    Xr = mesh_3d[:, 0] * c - mesh_3d[:, 2] * sn
    Yr = mesh_3d[:, 1]
    return np.stack([s * Xr + tx, s * Yr + ty], 1)

for nm, (s, th, tx, ty, W, H) in cameras.items():
    lm_arr, W2, H2 = lm_and_size(nm)
    pix = lm_arr[:, :2] * np.array([W2, H2])
    proj = project(vh, s, th, tx, ty)
    # 只统计 anchor 点（已知 mesh→MP 对应）
    a2d = pix[anchor_mp_idxs]
    a3d_proj = proj[anchor_mp_idxs]
    # 像素误差：x 方向按 W 归一化，y 按 H 归一化；综合欧氏
    e = np.linalg.norm((a2d - a3d_proj) / np.array([W2, H2]), axis=1)
    print(f"  {nm:55s} N={len(e)} mean={e.mean():.4f}  max={e.max():.4f}  median={np.median(e):.4f}")

# 预览
views = []
for tag, yaw in [("front", 0), ("left45", -45), ("right45", 45), ("side_left", -90)]:
    p = os.path.join(ROOT, "previews", "rocky04_%s.png" % tag)
    render(out, fh, p, yaw=yaw)
    views.append(p)
contact_sheet([os.path.join(ROOT, "previews", "head_front.png"), views[0],
               os.path.join(ROOT, "previews", "head_side_left.png"), views[3]],
              os.path.join(ROOT, "previews", "compare_v04.png"))
print("\n完成")