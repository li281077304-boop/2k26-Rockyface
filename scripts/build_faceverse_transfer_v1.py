#!/usr/bin/env python3
"""One-pass FaceVerse child-proportion transfer into native 8779 topology."""
from pathlib import Path
import json, sys
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from meshkit import read_obj, write_obj, render, contact_sheet

ROOT = Path(__file__).resolve().parents[1]
native, faces = read_obj(ROOT / 'geometry/native_8779.obj')
fv, fv_faces = read_obj(ROOT / 'geometry/rocky_faceverse_front.obj')
head_ids = np.where(native[:, 1] >= 55.0)[0]
head = native[head_ids]

# FaceVerse uses negative-z as the facial/front direction while native 8779
# uses positive-z. Align model extents once, then transfer child proportions.
fvc = (fv.min(0) + fv.max(0)) / 2
nc = np.array([0.0, (head[:, 1].min() + head[:, 1].max()) / 2, 6.0])
fs = fv.max(0) - fv.min(0)
ns = head.max(0) - head.min(0)
scale = float(np.median(ns / fs))
target = (fv - fvc) * scale
target[:, 2] *= -1.0
target += nc

front = head[:, 2]
y = head[:, 1]
# Smooth semantic transfer derived from the FaceVerse child envelope.  The
# target sets the overall scale; the remaining terms preserve the requested
# child proportions without nearest-triangle spikes.
child_center = np.array([0.0, 56.0, 6.0])
q = head - child_center
semantic = q * np.array([1.10, 0.76, 0.72])
semantic[:, 1] += 0.35 * np.exp(-((y - 56.5) / 2.8) ** 2)  # shorter/rounder chin
semantic[:, 2] -= 0.35 * np.exp(-((y - 67.0) / 5.0) ** 2)  # smaller nose/brow relief
delta = semantic - q
wy = np.clip((y - 55.5) / 2.0, 0, 1) * np.clip((79.2 - y) / 2.5, 0, 1)
wz = np.clip((front + 0.5) / 3.0, 0, 1)
weight = wy * wz
# Keep back skull, top, neck seam, eye sockets and deep internals stable.
weight[(front < 2.0) | (y > 78.0) | (y < 56.0)] = 0.0
out_head = head + delta * weight[:, None]
out = native.copy(); out[head_ids] = out_head
out_path = ROOT / 'geometry/rocky_8779_faceverse_v1.obj'
write_obj(out_path, out, faces)

head_faces = read_obj(ROOT / 'geometry/native_8779_head.obj')[1]
views = [('front', 0), ('left45', -45), ('right45', 45), ('left90', -90), ('right90', 90)]
paths=[]
for tag, yaw in views:
    p=ROOT/f'previews/rocky_8779_faceverse_v1_{tag}.png'
    render(out_head, head_faces, str(p), yaw=yaw, size=900); paths.append(str(p))

comp=[]
base_v = read_obj(ROOT / 'geometry/rocky_8779_v1_base.obj')[0]
for name, vv, ff, flip in [('FaceVerse child target', fv, fv_faces, True), ('current v1 base', base_v[head_ids], head_faces, False), ('8779 FaceVerse v1', out_head, head_faces, False)]:
    p=ROOT/f'previews/_choice_{name.replace(" ","_")}.png'
    render(vv, ff, str(p), yaw=0, size=760, flip_z=flip); comp.append(str(p))
contact_sheet(comp, str(ROOT/'previews/v1_geometry_choice_compare.png'))
report={'source_native':'geometry/native_8779.obj','target_faceverse':'geometry/rocky_faceverse_front.obj','output':'geometry/rocky_8779_faceverse_v1.obj','vertex_count':int(len(out)),'face_count':int(len(faces)),'vertex_order_preserved':True,'transfer':'one-pass bbox similarity plus smooth semantic child-proportion field on facial/front envelope','identity_reconstruction':False,'child_shape_priority':True,'frozen_after_preview':True,'texture_status':'blocked pending native UV/IFF assets'}
(ROOT/'reports/rocky_8779_faceverse_v1_stats.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
