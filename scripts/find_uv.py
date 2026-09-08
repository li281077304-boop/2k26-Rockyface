# -*- coding: utf-8 -*-
"""
定位 NBA 2K26 hihead 的 UV 流（程序化穷举，不靠肉眼猜）

已确认：
  LodVerts = 24484
  position buffer = VertexBuffer.f2fce607cb4c3b42.bin  stride 12 = float32[3]
  index buffer    = IndexBuffer.c829179a1bc03cf2.bin   uint16, 235236 个索引

只有 stride 能被 24484 整除的 buffer 才可能是 LOD0 的逐顶点流：
  f2fce607 293808/24484 = 12  -> 已确认是坐标
  7c6afba5 391744/24484 = 16  -> UV 最可能在这里
  其余两个不能整除，跳过。

对每个 (buffer, 字节偏移, 编码) 组合解码出 UV，按 6 项指标评分，
输出得分最高的若干候选的 UV wireframe 图。
"""
import zipfile, os, math
import numpy as np
from PIL import Image

IFF = r"D:\STEAM\steamapps\common\NBA 2K26\mods_work\RobbieAvila8779\png8779.iff"
OUT = os.path.dirname(os.path.abspath(__file__))
NV = 24484

z = zipfile.ZipFile(IFF)
pos = np.frombuffer(z.read("VertexBuffer.f2fce607cb4c3b42.bin"), dtype="<f4").reshape(-1, 3)
idx = np.frombuffer(z.read("IndexBuffer.c829179a1bc03cf2.bin"), dtype="<u2").reshape(-1, 3)
print("positions", pos.shape, " triangles", idx.shape)

# ---------- 导出 OBJ ----------
def write_obj(path, verts, tris):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# native 8779 hihead  verts=%d tris=%d\n" % (len(verts), len(tris)))
        for x, y, zz in verts:
            f.write("v %.6f %.6f %.6f\n" % (x, y, zz))
        for a, b, c in tris + 1:
            f.write("f %d %d %d\n" % (a, b, c))


write_obj(os.path.join(OUT, "native_8779.obj"), pos, idx.astype(np.int64))
print("已导出 native_8779.obj")

# ---------- 候选解码 ----------
def decode(buf, off, kind):
    n = len(buf)
    stride = n // NV
    if stride * NV != n:
        return None
    raw = np.frombuffer(buf, dtype=np.uint8).reshape(NV, stride)
    if kind == "f32x2":
        u = raw[:, off:off + 4].copy().view("<f4").ravel()
        v = raw[:, off + 4:off + 8].copy().view("<f4").ravel()
    elif kind == "f16x2":
        u = raw[:, off:off + 2].copy().view("<f2").ravel().astype(np.float64)
        v = raw[:, off + 2:off + 4].copy().view("<f2").ravel().astype(np.float64)
    elif kind == "unorm16x2":
        u = raw[:, off:off + 2].copy().view("<u2").ravel().astype(np.float64) / 65535.0
        v = raw[:, off + 2:off + 4].copy().view("<u2").ravel().astype(np.float64) / 65535.0
    elif kind == "snorm16x2":
        u = np.maximum(raw[:, off:off + 2].copy().view("<i2").ravel().astype(np.float64) / 32767.0, -1.0)
        v = np.maximum(raw[:, off + 2:off + 4].copy().view("<i2").ravel().astype(np.float64) / 32767.0, -1.0)
    elif kind == "unorm8x2":
        u = raw[:, off].astype(np.float64) / 255.0
        v = raw[:, off + 1].astype(np.float64) / 255.0
    elif kind == "snorm8x2":
        u = np.maximum(raw[:, off].view(np.int8).astype(np.float64) / 127.0, -1.0)
        v = np.maximum(raw[:, off + 1].view(np.int8).astype(np.float64) / 127.0, -1.0)
    elif kind == "r10g10b10a2":
        w = raw[:, off:off + 4].copy().view("<u4").ravel()
        u = (w & 0x3FF).astype(np.float64) / 1023.0
        v = ((w >> 10) & 0x3FF).astype(np.float64) / 1023.0
    elif kind == "r11g11b10":
        w = raw[:, off:off + 4].copy().view("<u4").ravel()
        u = (w & 0x7FF).astype(np.float64) / 2047.0
        v = ((w >> 11) & 0x7FF).astype(np.float64) / 2047.0
    else:
        return None
    return np.stack([u, v], axis=1)


SIZES = {"f32x2": 8, "f16x2": 4, "unorm16x2": 4, "snorm16x2": 4,
         "unorm8x2": 2, "snorm8x2": 2, "r10g10b10a2": 4, "r11g11b10": 4}


E3 = np.concatenate([np.stack([idx[:, 0], idx[:, 1]], 1),
                     np.stack([idx[:, 1], idx[:, 2]], 1),
                     np.stack([idx[:, 2], idx[:, 0]], 1)], 0)
D3 = np.linalg.norm(pos[E3[:, 0]] - pos[E3[:, 1]], axis=1)


def score(uv):
    u, v = uv[:, 0], uv[:, 1]
    if not (np.isfinite(u).all() and np.isfinite(v).all()):
        return None
    if u.std() < 1e-6 or v.std() < 1e-6:
        return None
    inr = float(((u > -0.05) & (u < 1.05) & (v > -0.05) & (v < 1.05)).mean())
    a, b, c = uv[idx[:, 0]], uv[idx[:, 1]], uv[idx[:, 2]]
    area = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                  (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) * 0.5
    nondeg = float((area > 1e-9).mean())
    gx = np.clip((u * 63).astype(int), 0, 63)
    gy = np.clip((v * 63).astype(int), 0, 63)
    occ = len(set(zip(gx.tolist(), gy.tolist()))) / 4096.0
    # 杀手判据：UV 边长 vs 3D 边长的对数相关性（真实展开必然强相关）
    duv = np.linalg.norm(uv[E3[:, 0]] - uv[E3[:, 1]], axis=1)
    m = (D3 > 1e-9) & (duv > 1e-12)
    if m.sum() < 1000:
        return None
    la, lb = np.log(D3[m]), np.log(duv[m])
    corr = float(np.corrcoef(la, lb)[0, 1]) if la.std() > 1e-9 and lb.std() > 1e-9 else 0.0
    return {"in_range": inr, "nondeg": nondeg, "occupancy": occ, "corr": corr,
            "score": max(corr, 0) * 0.6 + inr * 0.2 + nondeg * 0.1 + min(occ, 0.5) * 0.2}


results = []
for name in ["VertexBuffer.7c6afba5d91ba744.bin", "VertexBuffer.f2fce607cb4c3b42.bin"]:
    buf = z.read(name)
    stride = len(buf) // NV
    if stride * NV != len(buf):
        continue
    for kind, sz in SIZES.items():
        for off in range(0, stride - sz + 1):
            uv = decode(buf, off, kind)
            if uv is None:
                continue
            s = score(uv)
            if s is None:
                continue
            s.update({"buf": name[:24], "kind": kind, "off": off, "uv": uv})
            results.append(s)

results.sort(key=lambda r: -r["score"])
print("\n候选数 %d，得分前 12：" % len(results))
print("%-26s %-12s %3s  %7s %7s %7s %7s" % ("buffer", "encoding", "off", "in_rng", "nondeg", "occup", "score"))
for r in results[:12]:
    print("%-26s %-12s %3d  %7.3f %7.3f %7.3f %7.4f"
          % (r["buf"], r["kind"], r["off"], r["in_range"], r["nondeg"], r["occupancy"], r["score"]))


def render_uv(uv, path, size=1024, samples=6):
    """用三角形边采样画 UV wireframe"""
    img = np.zeros((size, size), bool)
    t = np.linspace(0, 1, samples)[:, None]
    for e in range(3):
        p0 = uv[idx[:, e]]
        p1 = uv[idx[:, (e + 1) % 3]]
        pts = p0[None, :, :] * (1 - t)[..., None] + p1[None, :, :] * t[..., None]
        xs = np.clip((pts[..., 0] * (size - 1)).astype(int), 0, size - 1).ravel()
        ys = np.clip((pts[..., 1] * (size - 1)).astype(int), 0, size - 1).ravel()
        img[ys, xs] = True
    out = np.where(img, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(path)


print("\n渲染前 6 名候选 wireframe ...")
for i, r in enumerate(results[:6]):
    p = os.path.join(OUT, "candidate_uv_%02d_%s_off%d.png" % (i + 1, r["kind"], r["off"]))
    render_uv(r["uv"], p)
    print("  ", os.path.basename(p))
