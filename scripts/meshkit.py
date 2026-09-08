# -*- coding: utf-8 -*-
"""轻量软件光栅化渲染器 + OBJ 读写（供 Rocky 几何预览用，不依赖 Blender）"""
import os
import numpy as np
from PIL import Image


# ---------------- OBJ 读写 ----------------
def read_obj(path):
    v, f = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                v.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:4]]
                f.append(idx)
    return np.array(v), np.array(f, dtype=np.int64)


def write_obj(path, verts, tris, uvs=None):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# verts=%d tris=%d\n" % (len(verts), len(tris)))
        for x, y, z in verts:
            fh.write("v %.6f %.6f %.6f\n" % (x, y, z))
        if uvs is not None:
            for u, w in uvs:
                fh.write("vt %.6f %.6f\n" % (u, w))
        if uvs is None:
            for a, b, c in tris + 1:
                fh.write("f %d %d %d\n" % (a, b, c))
        else:
            for (a, b, c) in tris + 1:
                fh.write("f %d/%d %d/%d %d/%d\n" % (a, a, b, b, c, c))


# ---------------- 光栅化渲染 ----------------
def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def render(verts, tris, out_path, yaw=0.0, pitch=0.0, size=760, flip_z=False,
           uv=None, tex=None):
    """正交投影 + 平面着色 + z-buffer。若给 uv+tex 则纹理采样。Z 轴朝向观察者。"""
    v = verts.copy()
    if flip_z:
        v[:, 2] = -v[:, 2]
    R = _rot("x", pitch) @ _rot("y", yaw)
    v = v @ R.T
    if uv is not None:
        uuv = uv.copy()
        if flip_z:
            pass
        uvR = uv.copy()

    lo, hi = v.min(0), v.max(0)
    span = max(hi[0] - lo[0], hi[1] - lo[1]) * 1.12
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    px = ((v[:, 0] - cx) / span + 0.5) * (size - 1)
    py = ((1 - (v[:, 1] - cy) / span) * (size - 1))
    pz = v[:, 2]

    tri = tris
    a, b, c = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1)
    ok = ln > 1e-12
    n[ok] /= ln[ok][:, None]

    light = np.array([0.25, 0.45, 0.86])
    light /= np.linalg.norm(light)
    lam = np.clip(n @ light, 0, 1)
    shade = (0.28 + 0.72 * lam)[:, None] * np.array([200, 205, 212])[None, :]
    depth = (pz[tri[:, 0]] + pz[tri[:, 1]] + pz[tri[:, 2]]) / 3
    order = np.argsort(-depth)

    if tex is not None and uv is not None:
        tw, th = tex.shape[1], tex.shape[0]
        ut = np.clip(uv[tri][:, :, 0], 0, 1) * (tw - 1)
        vt = np.clip(1 - uv[tri][:, :, 1], 0, 1) * (th - 1)

    img = np.full((size, size, 3), 246, np.uint8)
    zbuf = np.full((size, size), -1e9, np.float64)
    P = np.stack([px, py], 1).astype(np.float64)
    for t in order:
        if not ok[t]:
            continue
        p0, p1, p2 = P[tri[t, 0]], P[tri[t, 1]], P[tri[t, 2]]
        xmin = max(int(min(p0[0], p1[0], p2[0])), 0)
        xmax = min(int(max(p0[0], p1[0], p2[0])) + 1, size - 1)
        ymin = max(int(min(p0[1], p1[1], p2[1])), 0)
        ymax = min(int(max(p0[1], p1[1], p2[1])) + 1, size - 1)
        if xmax < xmin or ymax < ymin:
            continue
        d = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
        if abs(d) < 1e-12:
            continue
        xs = np.arange(xmin, xmax + 1)
        ys = np.arange(ymin, ymax + 1)
        gx, gy = np.meshgrid(xs, ys)
        w0 = ((p1[0] - gx) * (p2[1] - gy) - (p2[0] - gx) * (p1[1] - gy)) / d
        w1 = ((p2[0] - gx) * (p0[1] - gy) - (p0[0] - gx) * (p2[1] - gy)) / d
        w2 = 1 - w0 - w1
        m = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not m.any():
            continue
        zz = w0 * pz[tri[t, 0]] + w1 * pz[tri[t, 1]] + w2 * pz[tri[t, 2]]
        sub_z = zbuf[ymin:ymax + 1, xmin:xmax + 1]
        upd = m & (zz > sub_z)
        sub_z[upd] = zz[upd]
        col = shade[t]
        if tex is not None and uv is not None:
            tu = w0 * ut[t, 0] + w1 * ut[t, 1] + w2 * ut[t, 2]
            tv = w0 * vt[t, 0] + w1 * vt[t, 1] + w2 * vt[t, 2]
            xid = np.clip(tu[upd].astype(int), 0, tw - 1)
            yid = np.clip(tv[upd].astype(int), 0, th - 1)
            col = tex[yid, xid]
            img[ymin:ymax + 1, xmin:xmax + 1][upd] = col
        else:
            img[ymin:ymax + 1, xmin:xmax + 1][upd] = col
    Image.fromarray(img).save(out_path)
    return img


def render_pick(verts, tris, yaw=0.0, pitch=0.0, size=512):
    """正交投影渲染并返回 pick 缓冲：图像、每像素三角形id与重心坐标。
    返回 dict(png=uint8(H,W,3), tri=int32(H,W), w=float32(H,W,3), v=投影后坐标)"""
    v = verts.copy()
    R = _rot("x", pitch) @ _rot("y", yaw)
    v = v @ R.T
    lo, hi = v.min(0), v.max(0)
    span = max(hi[0] - lo[0], hi[1] - lo[1]) * 1.12
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    px = ((v[:, 0] - cx) / span + 0.5) * (size - 1)
    py = ((1 - (v[:, 1] - cy) / span) * (size - 1))
    pz = v[:, 2]

    tri = tris
    a, b, c = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1)
    ok = ln > 1e-12
    n[ok] /= ln[ok][:, None]
    light = np.array([0.25, 0.45, 0.86])
    light /= np.linalg.norm(light)
    lam = np.clip(n @ light, 0, 1)
    shade = (0.32 + 0.68 * lam)[:, None] * np.array([206, 210, 216])[None, :]
    order = np.argsort(-(pz[tri[:, 0]] + pz[tri[:, 1]] + pz[tri[:, 2]]) / 3)

    img = np.full((size, size, 3), 250, np.uint8)
    zbuf = np.full((size, size), -1e9, np.float64)
    ptri = np.full((size, size), -1, np.int32)
    pw = np.zeros((size, size, 3), np.float32)
    P = np.stack([px, py], 1).astype(np.float64)
    for t in order:
        if not ok[t]:
            continue
        p0, p1, p2 = P[tri[t, 0]], P[tri[t, 1]], P[tri[t, 2]]
        xmin = max(int(min(p0[0], p1[0], p2[0])), 0)
        xmax = min(int(max(p0[0], p1[0], p2[0])) + 1, size - 1)
        ymin = max(int(min(p0[1], p1[1], p2[1])), 0)
        ymax = min(int(max(p0[1], p1[1], p2[1])) + 1, size - 1)
        if xmax < xmin or ymax < ymin:
            continue
        d = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
        if abs(d) < 1e-12:
            continue
        xs = np.arange(xmin, xmax + 1); ys = np.arange(ymin, ymax + 1)
        gx, gy = np.meshgrid(xs, ys)
        w0 = ((p1[0] - gx) * (p2[1] - gy) - (p2[0] - gx) * (p1[1] - gy)) / d
        w1 = ((p2[0] - gx) * (p0[1] - gy) - (p0[0] - gx) * (p2[1] - gy)) / d
        w2 = 1 - w0 - w1
        m = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not m.any():
            continue
        zz = w0 * pz[tri[t, 0]] + w1 * pz[tri[t, 1]] + w2 * pz[tri[t, 2]]
        sub_z = zbuf[ymin:ymax + 1, xmin:xmax + 1]
        upd = m & (zz > sub_z)
        sub_z[upd] = zz[upd]
        img[ymin:ymax + 1, xmin:xmax + 1][upd] = shade[t]
        ptri[ymin:ymax + 1, xmin:xmax + 1][upd] = t
        pw[ymin:ymax + 1, xmin:xmax + 1][upd] = \
            np.stack([w0, w1, w2], 2)[upd]
    return {"img": img, "tri": ptri, "w": pw}


def pick_vertex(ptri, pw, tris, x, y):
    """由 pick 缓冲 + 像素坐标反查顶点 id（取重心权重最大角点）"""
    t = int(ptri[y, x])
    if t < 0:
        return None
    w = pw[y, x]
    k = int(np.argmax(w))
    return int(tris[t, k])


def four_views(verts, tris, prefix, flip_z=False):
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    render(verts, tris, prefix + "_front.png", yaw=0, flip_z=flip_z)
    render(verts, tris, prefix + "_left45.png", yaw=-45, flip_z=flip_z)
    render(verts, tris, prefix + "_right45.png", yaw=45, flip_z=flip_z)
    render(verts, tris, prefix + "_side_left.png", yaw=-90, flip_z=flip_z)


def contact_sheet(paths, out_path, labels=None):
    ims = [Image.open(p) for p in paths]
    w = max(i.width for i in ims)
    h = max(i.height for i in ims)
    cv = Image.new("RGB", (w * len(ims), h), (250, 250, 250))
    for k, im in enumerate(ims):
        cv.paste(im, (k * w, 0))
    cv.save(out_path)
