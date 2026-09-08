# -*- coding: utf-8 -*-
"""带原生贴图的头模渲染 + 检测调试"""
import sys, os, zipfile, io
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from meshkit import render
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
IFF = r"D:\STEAM\steamapps\common\NBA 2K26\mods_work\RobbieAvila8779\png8779.iff"
CFG = r"D:\STEAM\steamapps\common\NBA 2K26\mods_work\RobbieAvila8779\png8779_config_afro.iff"

z = zipfile.ZipFile(IFF)
zc = zipfile.ZipFile(CFG)
buf = z.read("VertexBuffer.7c6afba5d91ba744.bin")
uv_full = (np.frombuffer(buf, dtype=np.uint8).reshape(-1, 16)[:, 4:8]
           .copy().view("<u2").astype(np.float64) / 65535.0)
head_idx = np.load(os.path.join(ROOT, "geometry", "head_vertex_index.npy"))
uv_head = uv_full[head_idx]
tex = Image.open(io.BytesIO(zc.read("face_color_o.11b6e157b70d15d3.dds"))).convert("RGB")
print("head uv range", np.round(uv_head.min(0), 3), np.round(uv_head.max(0), 3))

vh = np.load(os.path.join(ROOT, "geometry", "native_8779_head_pos.npy"))
fh = np.load(os.path.join(ROOT, "geometry", "head_faces_local.npy"))
for tag, yaw in [("front", 0), ("side", -90)]:
    render(vh, fh, os.path.join(ROOT, "previews", "native_tex_%s.png" % tag),
           yaw=yaw, size=900, uv=uv_head, tex=np.asarray(tex))
print("done")
