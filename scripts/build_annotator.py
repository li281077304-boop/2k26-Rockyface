# -*- coding: utf-8 -*-
"""
生成 8779 头模 landmark 标注器（自包含单文件 HTML）
- 渲染 5 视角并生成每像素顶点 id 缓冲
- 将 图像 + 顶点id图 + 顶点坐标 内嵌为 base64
- 网页点选 ~45 个解剖点 → 导出 rocky_landmark_vertex_map.json

用法：浏览器打开 tools/annotator.html，点选 landmark 后按「导出 JSON」。
"""
import os, sys, json, base64, io
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from meshkit import render_pick  # noqa

SIZE = 512
vh = np.load(os.path.join(ROOT, "geometry", "native_8779_head_pos.npy"))
fh = np.load(os.path.join(ROOT, "geometry", "head_faces_local.npy"))

VIEWS = [("front", 0, 0), ("left45", -45, 0), ("right45", 45, 0),
         ("side_left", -90, 0), ("side_right", 90, 0)]

assets = {}  # view -> {png_b64, vid_b64}
for name, yaw, pitch in VIEWS:
    rp = render_pick(vh, fh, yaw=yaw, pitch=pitch, size=SIZE)
    img = rp["img"]
    tri = rp["tri"]
    w = rp["w"]
    # per-pixel vertex map
    vmap = np.zeros((SIZE, SIZE), np.uint16)
    vis = tri >= 0
    vmap[vis] = fh[tri[vis], np.argmax(w[vis], 1)]
    # save as PNG (uint16 grayscale) 便于浏览器直接解码? 用原始 bin+base64 更稳
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    png_b64 = base64.b64encode(buf.getvalue()).decode()
    vid_b64 = base64.b64encode(vmap.astype("<u2").tobytes()).decode()
    assets[name] = {"png": png_b64, "vid": vid_b64}
    print("view %-11s rendered %dx%d" % (name, SIZE, SIZE))

verts_b64 = base64.b64encode(vh.astype("<f4").tobytes()).decode()
meta = {"size": SIZE, "nvert": len(vh), "views": [v[0] for v in VIEWS]}

LANDMARKS = [
    ("nose_tip", "front"), ("nose_bridge", "front"),
    ("left_nostril", "front"), ("right_nostril", "front"),
    ("left_eye_outer", "front"), ("left_eye_inner", "front"),
    ("right_eye_outer", "front"), ("right_eye_inner", "front"),
    ("left_eye_upper", "front"), ("left_eye_lower", "front"),
    ("right_eye_upper", "front"), ("right_eye_lower", "front"),
    ("left_brow_in", "front"), ("left_brow_mid", "front"), ("left_brow_out", "front"),
    ("right_brow_in", "front"), ("right_brow_mid", "front"), ("right_brow_out", "front"),
    ("left_mouth_corner", "front"), ("right_mouth_corner", "front"),
    ("upper_lip", "front"), ("lower_lip", "front"),
    ("chin", "front"), ("chin_left", "front"), ("chin_right", "front"),
    ("left_jaw_mid", "front"), ("right_jaw_mid", "front"),
    ("left_cheek", "front"), ("right_cheek", "front"),
    ("forehead_mid", "front"), ("left_temple", "front"), ("right_temple", "front"),
    ("left_ear_top", "side_left"), ("left_ear_bottom", "side_left"),
    ("right_ear_top", "side_right"), ("right_ear_bottom", "side_right"),
    ("brow_profile", "side_left"), ("nose_profile", "side_left"),
    ("lip_profile", "side_left"), ("chin_profile", "side_left"),
]

HTML = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>8779 Head Landmark Annotator</title>
<style>
body{font-family:system-ui,'Microsoft YaHei';margin:0;background:#f4f4f6;color:#222}
h2{font-weight:500;margin:8px 14px}
#wrap{display:flex;gap:0}
#left{width:210px;padding:10px;background:#fff;border-right:1px solid #ddd;height:100vh;overflow:auto;box-sizing:border-box}
#lmList div{padding:3px 6px;margin:1px 0;border-radius:4px;cursor:pointer;font-size:13px;white-space:nowrap;overflow:hidden}
#lmList .done{background:#d8f0d8}.sel{outline:2px solid #2a7}
#main{flex:1;display:flex;flex-wrap:wrap;gap:8px;padding:10px}
.iv{position:relative;background:#fff;border:1px solid #ccc;border-radius:6px}
.iv .cap{position:absolute;left:6px;top:4px;background:rgba(255,255,255,.85);padding:1px 6px;border-radius:4px;font-size:12px}
canvas{display:block;cursor:crosshair}
.btns{position:fixed;bottom:8px;right:10px}
.btns button{margin-left:8px;padding:8px 16px;border:0;border-radius:6px;background:#2a7;color:#fff;font-size:14px;cursor:pointer}
#out{margin:4px 10px;font-family:monospace;font-size:12px;white-space:pre-wrap;max-height:120px;overflow:auto;background:#fff;border:1px solid #ddd;border-radius:4px;padding:6px;display:none}
</style></head><body>
<h2>8779 头模 landmark 标定 <span style="font-size:12px;color:#777">先在左侧选 landmark（绿色=已标），再到图上点击。一个点可在多个视角补标。</span></h2>
<div id="wrap">
<div id="left"><div id="lmList"></div></div>
<div id="main"></div>
</div>
<div class="btns">
<button onclick="saveJson()">导出 JSON</button>
<button onclick="clearOne()">清除当前</button>
<button onclick="prev();next()" ></button>
</div>
<div id="out"></div>
<script>
const META=__META__;
const VERTS=__VERTS__;      // Float32Array
const VIEWB=__VIEWB__;      // {view: {png:dataurl, vid:Uint16Array}}
const LAND=__LAND__;        // [[name, defaultView],...]
let marks={};               // name -> {x:{view:[x,y,vid]}}
let cur=0, curView='front';

function b64arr(b64){const s=atob(b64),u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return u;}
const vm={}; for(const v of Object.keys(VIEWB)){vm[v]=new Uint16Array(b64arr(VIEWB[v].vid).buffer);}

const main=document.getElementById('main');
const canvases={};
function buildViews(){
  for(const v of META.views){
    const iv=document.createElement('div');iv.className='iv';
    const cap=document.createElement('div');cap.className='cap';cap.textContent=v;
    const c=document.createElement('canvas');c.width=META.size;c.height=META.size;
    const img=new Image();img.onload=()=>c.getContext('2d').drawImage(img,0,0);
    img.src='data:image/png;base64,'+VIEWB[v].png;
    c.addEventListener('click',e=>{clickOn(v,e)});
    iv.appendChild(cap);iv.appendChild(c);main.appendChild(iv);canvases[v]=c;
  }
}
function drawDots(view){
  const c=canvases[view],g=c.getContext('2d');
  g.clearRect(0,0,c.width,c.height);
  const img=g.getImageData(0,0,c.width,c.height);g.putImageData(img,0,0);
  // 重绘图片
  const d=new Image();d.onload=()=>{g.clearRect(0,0,c.width,c.height);g.drawImage(d,0,0);
    for(const name in marks){for(const vv in marks[name]){if(vv!==view)continue;
      const [x,y]=marks[name][vv];g.fillStyle=name===LAND[cur][0]?'#ff0':'#2a7';
      g.beginPath();g.arc(x,y,4,0,7);g.fill();}}
  };d.src='data:image/png;base64,'+VIEWB[view].png;
}
function refresh(){drawDots(curView);refreshList();}
function refreshList(){const box=document.getElementById('lmList');box.innerHTML='';
  LAND.forEach(([nm],i)=>{const d=document.createElement('div');d.textContent=(i)+' '+nm+(marks[nm]?' ✓':'');
    d.className=(i===cur?'sel':'')+(marks[nm]?' done':'');d.onclick=()=>{cur=i;refresh();};box.appendChild(d);});}
function clickOn(view,e){
  const r=canvases[view].getBoundingClientRect();
  const x=Math.round((e.clientX-r.left)*(META.size/r.width));
  const y=Math.round((e.clientY-r.top)*(META.size/r.height));
  if(x<0||y<0||x>=META.size||y>=META.size)return;
  const vid=vm[view][y*META.size+x];
  if(vid>=META.nvert)return;
  const [nm]=LAND[cur];
  if(!marks[nm])marks[nm]={};
  marks[nm][view]=[x,y,vid];
  cur=(cur+1)%LAND.length;
  curView=view;
  refresh();
}
function clearOne(){const nm=LAND[cur][0];delete marks[nm];refresh();}
function prev(){cur=(cur+LAND.length-1)%LAND.length;refresh();}
function next(){cur=(cur+1)%LAND.length;refresh();}
function saveJson(){
  const out={};
  for(const [nm] of LAND){ if(!marks[nm])continue;
    const src=marks[nm]; let sx=0,sy=0,sz=0,n=0;
    const per={};
    for(const v in src){const vid=src[v][2];const o=vid*3;
      sx+=VERTS[o];sy+=VERTS[o+1];sz+=VERTS[o+2];n++;
      per[v]={x:src[v][0],y:src[v][1],vertex:vid};
    }
    out[nm]={pos:[+(sx/n).toFixed(3),+(sy/n).toFixed(3),+(sz/n).toFixed(3)],sources:per};
  }
  const txt=JSON.stringify(out,null,1);
  document.getElementById('out').style.display='block';
  document.getElementById('out').textContent=txt;
  if(navigator.clipboard) navigator.clipboard.writeText(txt).then(()=>{});
}
buildViews();refreshList();
window.addEventListener('keydown',e=>{
  if(e.key==='ArrowRight')next();if(e.key==='ArrowLeft')prev();
});
</script></body></html>
"""

html = (HTML
        .replace("__META__", json.dumps(meta))
        .replace("__VERTS__", "(()=>{const u=b64arr('%s');return new Float32Array(u.buffer);})()" % verts_b64)
        .replace("__VIEWB__", "(()=>{" + "return {" +
                 ",".join('"%s":{png:%s,vid:%s}' % (
                     v, json.dumps(assets[v]["png"]), json.dumps(assets[v]["vid"]))
                     for v in assets) + "};})()")
        .replace("__LAND__", json.dumps(LANDMARKS)))

outdir = os.path.join(ROOT, "tools")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "annotator.html")
with open(path, "w", encoding="utf-8") as f:
    f.write(html)
print("生成:", path, "大小 %.1f MB" % (os.path.getsize(path) / 1e6))
