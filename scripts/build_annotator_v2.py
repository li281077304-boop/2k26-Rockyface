# -*- coding: utf-8 -*-
"""
8779 头模 landmark 标注器 v2
- 800px 渲染 + 每像素顶点id缓冲 + 顶点镜像映射 + 每视角顶点像素表
- 单文件 HTML：缩放/拖动/撤销/镜像自动/中文说明/localStorage 自动保存
- 人工只标 ~26 点；镜像对自动补另一侧（可修正）
"""
import os, sys, json, base64, io
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from meshkit import render_pick  # noqa

SIZE = 800
vh = np.load(os.path.join(ROOT, "geometry", "native_8779_head_pos.npy"))
fh = np.load(os.path.join(ROOT, "geometry", "head_faces_local.npy"))
N = len(vh)

VIEWS = [("front", 0, 0), ("left45", -45, 0), ("right45", 45, 0),
         ("side_left", -90, 0), ("side_right", 90, 0)]

# ------- 镜像映射：每顶点 x 取反后的最近顶点 -------
print("计算镜像映射 ...")
tree = cKDTree(vh[:, [0, 1, 2]])
ref = vh.copy(); ref[:, 0] = -ref[:, 0]
_, mirror_idx = tree.query(ref)
print("镜像顶点映射完成")

# ------- 渲染与缓冲 -------
def b64(b):
    return base64.b64encode(b).decode()


VIEWB = {}
for name, yaw, pitch in VIEWS:
    rp = render_pick(vh, fh, yaw=yaw, pitch=pitch, size=SIZE)
    tri = rp["tri"]; w = rp["w"]
    vmap = np.full((SIZE, SIZE), 65535, np.uint16)
    vis = tri >= 0
    vmap[vis] = fh[tri[vis], np.argmax(w[vis], 1)]
    # 每视角顶点→像素
    v = vh.copy()
    lo, hi = v.min(0), v.max(0)
    span = max(hi[0] - lo[0], hi[1] - lo[1]) * 1.12
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    R = None
    # 复用 render_pick 内投影: 直接重算像素
    import math
    rad = math.radians(yaw)
    c, s = math.cos(rad), math.sin(rad)
    vv = v @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]).T
    px = ((vv[:, 0] - cx) / span + 0.5) * (SIZE - 1)
    py = (1 - (vv[:, 1] - cy) / span) * (SIZE - 1)
    vpix = np.stack([px, py], 1).astype("<f4")
    img = rp["img"]
    from PIL import Image
    buf = io.BytesIO(); Image.fromarray(img).save(buf, format="PNG")
    VIEWB[name] = {"png": b64(buf.getvalue()), "vid": b64(vmap.astype("<u2").tobytes()),
                   "vpix": b64(vpix.tobytes())}
    print("view %-10s ok" % name)

# ------- landmark 清单 -------
# (key, 中文名, 提示, 镜像key|None, 主要视图)
L = [
    ("left_eye_outer", "左外眼角", "眼睛最外侧角。标完自动补右外眼角。", "right_eye_outer", "front"),
    ("left_eye_inner", "左内眼角", "眼睛最内侧(靠鼻)角。", "right_eye_inner", "front"),
    ("left_brow_in", "左眉头", "眉毛靠近鼻子的起点。", "right_brow_in", "front"),
    ("left_brow_arch", "左眉峰", "眉毛最高/弯点。", "right_brow_arch", "front"),
    ("left_nostril", "左鼻翼", "鼻孔外侧边缘。", "right_nostril", "front"),
    ("left_mouth_corner", "左嘴角", "嘴缝最左端。", "right_mouth_corner", "front"),
    ("left_jaw_angle", "左下颌角", "下颌骨拐角(耳下前方)。", "right_jaw_angle", "front"),
    ("left_cheekbone", "左颧骨", "颧骨外缘最高点(眼下外侧)。", "right_cheekbone", "front"),
    ("nose_root", "鼻根", "两眼之间、眉心下凹陷。正面标, 侧面补一次。", None, "front"),
    ("nose_tip", "鼻尖", "正面标; 请再到侧视图补标一次确定前突。", None, "front"),
    ("upper_lip", "上唇中心", "上唇人中下缘中点。", None, "front"),
    ("lower_lip", "下唇中心", "下唇最下缘中点。", None, "front"),
    ("chin", "下巴尖", "正面标, 侧面补一次确定前突。", None, "front"),
    ("left_ear_top", "左耳顶", "换到 side_left 视图标左耳顶端。", None, "side_left"),
    ("left_ear_bottom", "左耳底", "换到 side_left 视图标左耳垂底。", None, "side_left"),
]
# 右侧镜像自动由左侧补出；右耳在 side_right 标
R = [
    ("right_eye_outer", "右外眼角", "自动镜像左外眼角; 若不准请点击修正。", None, "front"),
    ("right_eye_inner", "右内眼角", "自动镜像。", None, "front"),
    ("right_brow_in", "右眉头", "自动镜像。", None, "front"),
    ("right_brow_arch", "右眉峰", "自动镜像。", None, "front"),
    ("right_nostril", "右鼻翼", "自动镜像。", None, "front"),
    ("right_mouth_corner", "右嘴角", "自动镜像。", None, "front"),
    ("right_jaw_angle", "右下颌角", "自动镜像。", None, "front"),
    ("right_cheekbone", "右颧骨", "自动镜像。", None, "front"),
    ("right_ear_top", "右耳顶", "在 side_right 视图标。", None, "side_right"),
    ("right_ear_bottom", "右耳底", "在 side_right 视图标。", None, "side_right"),
]
LAND = L + R
# 每个左侧点的镜像源映射
mirror_src = {}
for k, zh, hint, mk, view in L:
    if mk:
        mirror_src[mk] = k

HTML_TMPL = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>8779 Head Landmark Annotator v2</title>
<style>
*{box-sizing:border-box}body{font-family:system-ui,'Microsoft YaHei';margin:0;background:#eef0f2;color:#1d1f22}
#hd{padding:10px 14px;background:#fff;border-bottom:1px solid #ddd;position:sticky;top:0;z-index:5}
#hd h2{display:inline;font-size:17px;font-weight:500}
#wrap{display:flex}
#side{width:270px;background:#fff;border-right:1px solid #ddd;padding:8px;height:calc(100vh - 54px);overflow:auto}
#info{background:#f5f8ff;border:1px solid #bcd;border-radius:6px;padding:8px;margin-bottom:8px;font-size:13px}
#info b{display:block;font-size:14px}
#lmList div{padding:4px 8px;margin:1px;border-radius:5px;font-size:13px;cursor:pointer;border:1px solid transparent}
#lmList .done{background:#e4f6e4}#lmList .sel{border-color:#2a7;background:#eafbea}
#grid{flex:1;display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:8px;padding:10px;align-content:start}
.iv{position:relative;background:#111;border-radius:8px;overflow:hidden;touch-action:none}
.iv .cap{position:absolute;left:6px;top:5px;background:rgba(0,0,0,.55);color:#fff;padding:2px 8px;border-radius:10px;font-size:11px;z-index:3;pointer-events:none}
.iv canvas{position:absolute;left:0;top:0;cursor:crosshair}
#foot{position:fixed;bottom:8px;right:10px;z-index:6}
#foot button{margin-left:6px;padding:7px 14px;border:0;border-radius:6px;font-size:13px;cursor:pointer}
.b1{background:#2a7;color:#fff}.b2{background:#fff;border:1px solid #bbb!important;color:#333}
#tips{font-size:12px;color:#666;padding:2px 14px 6px}
#out{margin:8px 14px;font-family:ui-monospace,monospace;font-size:11px;white-space:pre;max-height:200px;overflow:auto;background:#0d1117;color:#9ecbff;border-radius:6px;padding:8px;display:none;position:fixed;bottom:52px;left:14px;right:280px;z-index:6}
</style></head><body>
<div id="hd"><h2>8779 头模 landmark 标注 <span id="progress"></span></h2>
<span style="font-size:12px;color:#888;margin-left:14px">滚轮缩放 · 拖动平移 · 双击复位 · ←/→ 切换 · Z 撤销 · 空格 自动下一项</span></div>
<div id="tips">标点顺序建议：先在正面把眼/眉/鼻/嘴/颧骨/下颌标完（左侧栏会高亮下一个）；<b>镜像点会自动补另一侧</b>，若位置不准点一下正确位置即可修正。随后按提示到 side_left / side_right 标双耳。鼻尖/下巴可在侧视图补标一次定深度。数据自动保存在浏览器，随时可关页面。</div>
<div id="wrap">
<div id="side"><div id="info"></div><div id="lmList"></div></div>
<div id="grid"></div>
</div>
<div id="out"></div>
<div id="foot"><button class="b2" onclick="undo()">撤销 Z</button><button class="b2" onclick="prevL()">← 上一点</button>
<button class="b1" onclick="nextL()">下一点 →</button><button class="b1" onclick="exportJ()">导出 JSON</button></div>
<script>
const META={size:800,nvert:__N__,views:["front","left45","right45","side_left","side_right"]};
const VERTS=__VERTS__, MIRROR=__MIRROR__, FIRSTTRI=__FIRSTTRI__;
const VIEWB=__VIEWB__;
const LAND=__LAND__;
const MSRC=__MSRC__;
const STORE='rk_annot_v2';
let marks={},cur=0,curView='front';
function b64b(b64){const s=atob(b64),u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return u;}
function arrU16(b){return new Uint16Array(b64b(b).buffer);}
function arrF32(b){return new Float32Array(b64b(b).buffer);}
const vm={},vp={};for(const v of META.views){vm[v]=arrU16(VIEWB[v].vid);vp[v]=arrF32(VIEWB[v].vpix);}
const cvs={},st={}; // state {sc,tx,ty}
function buildViews(){
  const grid=document.getElementById('grid');
  for(const v of META.views){
    const iv=document.createElement('div');iv.className='iv';
    const cap=document.createElement('div');cap.className='cap';cap.textContent=v;
    const c=document.createElement('canvas');c.width=META.size;c.height=META.size;
    iv.appendChild(cap);iv.appendChild(c);grid.appendChild(iv);cvs[v]=c;
    st[v]={sc:1,tx:0,ty:0};
    const img=new Image();
    img.onload=()=>{c.dataset.loaded='1';drawView(v);};
    img.src='data:image/png;base64,'+VIEWB[v].png;
    c._img=img;
    iv.addEventListener('wheel',e=>{e.preventDefault();const r=iv.getBoundingClientRect();
      const s=st[v],nx=e.clientX-r.left,ny=e.clientY-r.top;
      const ns=Math.clamp(s.sc*Math.exp(-e.deltaY*0.001),0.5,12);
      s.tx=nx-(nx-s.tx)*ns/s.sc;s.ty=ny-(ny-s.ty)*ns/s.sc;s.sc=ns;drawView(v);},{passive:false});
    let drag=null;
    iv.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY};iv.setPointerCapture(e.pointerId);});
    iv.addEventListener('pointermove',e=>{if(!drag)return;const s=st[v];
      s.tx+=e.clientX-drag.x;s.ty+=e.clientY-drag.y;drag={x:e.clientX,y:e.clientY};drawView(v);});
    iv.addEventListener('pointerup',e=>{drag=null;});
    iv.addEventListener('dblclick',e=>{st[v]={sc:1,tx:0,ty:0};drawView(v);});
    c.addEventListener('click',e=>{
      const r=iv.getBoundingClientRect(),s=st[v];
      const x=Math.round((e.clientX-r.left-s.tx)/s.sc),y=Math.round((e.clientY-r.top-s.ty)/s.sc);
      if(x<0||y<0||x>=META.size||y>=META.size)return;
      pick(v,x,y);});
  }
}
Math.clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
function drawView(v){
  const c=cvs[v];if(!c||!c._img)return;const g=c.getContext('2d');
  g.setTransform(1,0,0,1,0,0);g.clearRect(0,0,META.size,META.size);
  const s=st[v];g.fillStyle='#111';g.fillRect(0,0,META.size,META.size);
  g.setTransform(s.sc,0,0,s.sc,s.tx,s.ty);
  g.drawImage(c._img,0,0);
  const selKey=LAND[cur][0];
  for(const name in marks)for(const vv in marks[name]){
    if(vv!==v)continue;
    const d=marks[name][vv];
    g.fillStyle=(name===selKey)?'#ffd400':'#35d07a';
    g.beginPath();g.arc(d[0],d[1],d[2]?5:4,0,7);g.fill();
    if(name===selKey){g.strokeStyle='#fff';g.lineWidth=1;g.beginPath();g.arc(d[0],d[1],9,0,7);g.stroke();}
  }
}
function refreshAll(){for(const v of META.views)drawView(v);refreshList();}
function refreshList(){
  const box=document.getElementById('lmList');box.innerHTML='';
  LAND.forEach((lm,i)=>{const d=document.createElement('div');
    d.textContent=(marks[lm[0]]?'✓ ':'')+(i)+' '+lm[1]+(lm[4]!=='front'&&lm[4]!=='side_left'?' *':'');
    d.className=(i===cur?'sel ':'')+(marks[lm[0]]?'done':'');
    d.onclick=()=>{cur=i;refreshAll();};box.appendChild(d);});
  const sel=LAND[cur];
  document.getElementById('progress').textContent=(' '+(cur+1)+'/'+LAND.length);
  document.getElementById('info').innerHTML='<b>'+sel[0]+'（'+sel[1]+'）</b>'+sel[2]+
    '<div style="margin-top:4px;color:#666">主视图:'+sel[4]+' | 镜像: '+(sel[3]||'无')+'</div>';
}
function setMark(name,view,x,y,auto){
  if(!marks[name])marks[name]={};
  marks[name][view]=[x,y,0];
  pushUndo(name,view);
  save();
}
function pick(view,x,y){
  const vid=vm[view][y*META.size+x];
  if(vid>=META.nvert)return;
  const [key,zh]=LAND[cur];
  if(marks[key]&&marks[key][view]){marks[key][view][0]=x;marks[key][view][1]=y;pushUndo(key,view);save();refreshAll();return;}
  setMark(key,view,x,y,false);
  // 若当前点是一个"镜像目标"的源(左侧)，且目标还没标 → 自动补另一侧(仅正面/45°)
  for(const k2 in MSRC){ if(MSRC[k2]===key&&!marks[k2]&&view!=='side_left'&&view!=='side_right'){
      const mvid=MIRROR[vid]; if(mvid>=META.nvert)continue;
      const px=vp[view][mvid*2],py=vp[view][mvid*2+1];
      if(px>=0&&px<META.size&&py>=0&&py<META.size){setMark(k2,view,Math.round(px),Math.round(py),true);}
    }}
  refreshAll();
}
const UNDO=[];
function pushUndo(n,v){UNDO.push([n,v]);if(UNDO.length>200)UNDO.shift();}
function undo(){const t=UNDO.pop();if(!t)return;const[n,v]=t;if(marks[n]&&marks[n][v])delete marks[n][v];
  if(marks[n]&&!Object.keys(marks[n]).length)delete marks[n];refreshAll();}
function save(){try{localStorage.setItem(STORE,JSON.stringify(marks));}catch(e){}}
function load(){try{const s=localStorage.getItem(STORE);if(s){marks=JSON.parse(s);}}catch(e){}}
function prevL(){cur=(cur+LAND.length-1)%LAND.length;refreshAll();}
function nextL(){cur=(cur+1)%LAND.length;refreshAll();}
function exportJ(){
  const miss=LAND.filter(lm=>!marks[lm[0]]).map(lm=>lm[1]);
  if(miss.length&&!confirm('还有 '+miss.length+' 个未标:\n'+miss.join('、')+'\n\n仍要导出?'))return;
  const out={};
  for(const[mn] of LAND){if(!marks[mn])continue;const src=marks[mn];let sx=0,sy=0,sz=0,n=0;const per={};
    for(const v in src){
      const vid2=vm[v][src[v][1]*META.size+src[v][0]];
      if(vid2===undefined||vid2>=META.nvert)continue;
      const o=vid2*3;sx+=VERTS[o];sy+=VERTS[o+1];sz+=VERTS[o+2];n++;
      per[v]={x:src[v][0],y:src[v][1],vertex:vid2};
    }
    if(!n)continue;
    out[mn]={pos:[+(sx/n).toFixed(3),+(sy/n).toFixed(3),+(sz/n).toFixed(3)],sources:per};
  }
  // vertex: pos 最近顶点
  const txt=JSON.stringify(out,null,1);
  const o=document.getElementById('out');o.style.display='block';o.textContent=txt;
  try{navigator.clipboard.writeText(txt);}catch(e){}
  // 尝试下载
  try{const b=new Blob([txt],{type:'application/json'});const a=document.createElement('a');
    a.href=URL.createObjectURL(b);a.download='rocky_landmark_vertex_map.json';a.click();}catch(e){}
}
function nearestVid(view,x,y){return vm[view][Math.round(y)*META.size+Math.round(x)];}
load();buildViews();refreshAll();
window.addEventListener('keydown',e=>{
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT')return;
  if(e.key==='ArrowRight')nextL();else if(e.key==='ArrowLeft')prevL();
  else if(e.key==='z'||e.key==='Z')undo();else if(e.key===' ') {e.preventDefault();nextL();}
});
</script></body></html>"""

# vertex->任意三角形 表 (标注时显示用)
firsttri = np.full(N, -1, np.int32)
for t in range(len(fh)):
    for k in fh[t]:
        if firsttri[k] < 0:
            firsttri[k] = t

data = dict(
    N="%d" % N,
    VERTS="(()=>{const u=b64b('%s');return new Float32Array(u.buffer);})()" % b64(vh.astype("<f4").tobytes()),
    MIRROR="(()=>{const u=b64b('%s');return new Uint16Array(u.buffer);})()" % b64(mirror_idx.astype("<u2").tobytes()),
    FIRSTTRI="(()=>{const u=b64b('%s');return new Int32Array(u.buffer);})()" % b64(firsttri.tobytes()),
    VIEWB="(()=>{" + "return{" + ",".join('"%s":{png:%s,vid:%s,vpix:%s}' % (
        v, json.dumps(VIEWB[v]["png"]), json.dumps(VIEWB[v]["vid"]), json.dumps(VIEWB[v]["vpix"]))
        for v in VIEWB) + "};})()",
    LAND=json.dumps(LAND, ensure_ascii=False),
    MSRC=json.dumps(mirror_src),
)
html = HTML_TMPL
for k, v in data.items():
    html = html.replace("__" + k + "__", v)

outdir = os.path.join(ROOT, "tools")
os.makedirs(outdir, exist_ok=True)
p = os.path.join(outdir, "annotator_v2.html")
with open(p, "w", encoding="utf-8") as f:
    f.write(html)
print("生成", p, "%.1f MB" % (os.path.getsize(p) / 1e6))
