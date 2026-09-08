/* Rocky 8779 Landmark Annotator v3 - directory/localhost edition */
"use strict";
var SCHEMA = 3;
var LOG = 800;              // 头模逻辑画布尺寸（世界坐标 0..800）
var MANIFEST = null, VERTS = null, MIRROR = null, FIRSTTRI = null, LAND = null, MSRC = null;
var marks = {};
var cur = 0, curView = "front", undoStack = [];
var cv = null, ctx = null, stage = null;
var loadedViews = {};       // view -> {img, vid:Uint16Array, vpix:Float32Array}
var st = { sc: 1, tx: 0, ty: 0 };
var STORE = "rk_annot_v3";

function $(id) { return document.getElementById(id); }
function log(txt, isErr) {
  var b = $("bootlog");
  if (!b) return;
  b.innerHTML += (isErr ? '<span class="err">[ERROR] ' : "[OK] ") + txt + "\n";
  if (isErr) { var e2 = $("errlog"); if (e2) { e2.style.display = "block"; e2.textContent += txt + "\n"; } }
}
window.addEventListener("error", function (e) {
  log("window.onerror: " + e.message + (e.filename ? " @" + e.filename.split("/").pop() + ":" + e.lineno : ""), true);
});
window.addEventListener("unhandledrejection", function (e) {
  log("unhandledrejection: " + (e.reason && e.reason.message ? e.reason.message : String(e.reason)), true);
});

function fetchBin(url) {
  return fetch(url).then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status + " " + url); return r.arrayBuffer(); });
}
function fetchJSON(url) {
  return fetch(url).then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status + " " + url); return r.json(); });
}

function boot() {
  log("HTML loaded");
  log("app.js loaded");
  try {
    var v = localStorage.getItem("annotator_version");
    if (v !== String(SCHEMA)) {
      Object.keys(localStorage).forEach(function (k) {
        if (k.indexOf("rk_annot") === 0 || k.indexOf("annotator_") === 0) localStorage.removeItem(k);
      });
      localStorage.setItem("annotator_version", String(SCHEMA));
      log("localStorage 已按 schema v" + SCHEMA + " 清理");
    } else {
      var saved = localStorage.getItem(STORE);
      if (saved) { try { marks = JSON.parse(saved); } catch (e) { marks = {}; } }
      log("localStorage 已恢复");
    }
  } catch (e) { log("localStorage 不可用: " + e.message, true); }

  return Promise.resolve()
    .then(function () {
      return fetchJSON("data/manifest.json").then(function (m) { MANIFEST = m; log("manifest loaded"); });
    })
    .then(function () {
      return fetchJSON("data/landmarks.json").then(function (m) { LAND = m.list; MSRC = m.mirror_source; log("landmark definitions loaded (" + LAND.length + ")"); });
    })
    .then(function () { return fetchBin("data/vertices.bin").then(function (b) { VERTS = new Float32Array(b); log("vertices loaded"); }); })
    .then(function () { return fetchBin("data/mirror.bin").then(function (b) { MIRROR = new Uint16Array(b); log("mirror map loaded"); }); })
    .then(function () { return fetchBin("data/firsttri.bin").then(function (b) { FIRSTTRI = new Int32Array(b); log("triangle map loaded"); }); })
    .then(function () { return loadView("front", true); })
    .then(function () { initUI(); log("canvas initialized"); $("boot").style.display = "none"; $("app").style.display = "block";
                        document.title = "READY"; document.body.dataset.ready = "1"; })
    .catch(function (e) { log((e && e.stack) ? e.stack : String(e), true); });
}

function loadView(vname, keep) {
  if (loadedViews[vname]) { curView = vname; syncView(); return Promise.resolve(); }
  var v = null;
  for (var i = 0; i < MANIFEST.views.length; i++) if (MANIFEST.views[i].name === vname) v = MANIFEST.views[i];
  if (!v) return Promise.reject(new Error("view not found " + vname));
  var rec = {}, img = new Image();
  return new Promise(function (resolve, reject) {
    img.onload = function () { rec.img = img; resolve(); };
    img.onerror = function () { reject(new Error("image load failed: " + v.img)); };
    img.src = "data/" + v.img;
  }).then(function () {
    return fetchBin("data/" + v.vid).then(function (b) { rec.vid = new Uint16Array(b); });
  }).then(function () {
    return fetchBin("data/" + v.vpix).then(function (b) { rec.vpix = new Float32Array(b); });
  }).then(function () {
    loadedViews[vname] = rec;
    curView = vname;
    if (!keep) {
      Object.keys(loadedViews).forEach(function (k) { if (k !== vname && k !== "front") { loadedViews[k] = null; delete loadedViews[k]; } });
    }
    log(vname + " image + buffers loaded");
    syncView();
  });
}

function initUI() {
  cv = $("cv"); ctx = cv.getContext("2d"); stage = document.querySelector(".stage");
  var vw = $("views");
  MANIFEST.views.forEach(function (v) {
    var b = document.createElement("button");
    b.textContent = v.name;
    b.onclick = function () { selectView(v.name); };
    vw.appendChild(b);
  });
  refreshList();
  stage.addEventListener("wheel", onWheel, { passive: false });
  stage.addEventListener("pointerdown", onDown);
  stage.addEventListener("pointermove", onMove);
  stage.addEventListener("pointerup", function () { drag = null; });
  stage.addEventListener("dblclick", function () { fitStage(); });
  stage.addEventListener("click", onPick);
  window.addEventListener("keydown", onKey);
  window.addEventListener("resize", fitStage);
  fitStage();
}
function selectView(vname) {
  curView = vname;
  var all = document.querySelectorAll("#views button");
  all.forEach(function (b) { b.className = b.textContent === curView ? "act" : ""; });
  $("cap").textContent = curView;
  if (!loadedViews[vname]) { loadView(vname, false).catch(function (e) { log(e.message, true); }); }
  else { syncView(); }
}
function syncView() { drawView(); }
function fitStage() {
  if (!cv || !stage) return;
  var w = stage.clientWidth, h = stage.clientHeight;
  if (!w || !h) return;
  var K = Math.max(0.2, Math.min(w, h) / LOG * 0.94);
  cv.width = w; cv.height = h;
  st = { sc: K, tx: (w - LOG * K) / 2, ty: (h - LOG * K) / 2 };
  drawView();
}
function drawView() {
  if (!cv || !loadedViews[curView] || !ctx) return;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#101112"; ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.setTransform(st.sc, 0, 0, st.sc, st.tx, st.ty);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(loadedViews[curView].img, 0, 0);
  var rec = loadedViews[curView];
  var selKey = LAND[cur][0];
  var R = 6 / st.sc;
  for (var name in marks) {
    var src = marks[name];
    if (!src[curView]) continue;
    var vid = src[curView].vertex;
    if (vid === undefined || vid >= MANIFEST.nvert) continue;
    var px = rec.vpix[vid * 2], py = rec.vpix[vid * 2 + 1];
    if (px < 0 || py < 0 || px >= LOG || py >= LOG) continue;
    ctx.fillStyle = (name === selKey) ? "#ffd400" : "#35d07a";
    ctx.beginPath(); ctx.arc(px, py, R, 0, 7); ctx.fill();
    if (name === selKey) { ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5 / st.sc; ctx.beginPath(); ctx.arc(px, py, R * 1.9, 0, 7); ctx.stroke(); }
  }
}
function worldXY(e) {
  var r = cv.getBoundingClientRect();
  return { x: (e.clientX - r.left - st.tx) / st.sc, y: (e.clientY - r.top - st.ty) / st.sc };
}
function onWheel(e) {
  e.preventDefault();
  var r = cv.getBoundingClientRect();
  var nx = e.clientX - r.left, ny = e.clientY - r.top;
  var ns = Math.max(0.2, Math.min(16, st.sc * Math.exp(-e.deltaY * 0.0015)));
  st.tx = nx - (nx - st.tx) * ns / st.sc;
  st.ty = ny - (ny - st.ty) * ns / st.sc;
  st.sc = ns; drawView();
}
var drag = null;
function onDown(e) { drag = { x: e.clientX, y: e.clientY }; stage.setPointerCapture(e.pointerId); }
function onMove(e) {
  if (!drag) return;
  st.tx += e.clientX - drag.x; st.ty += e.clientY - drag.y;
  drag = { x: e.clientX, y: e.clientY }; drawView();
}
function onPick(e) {
  if (!loadedViews[curView]) return;
  var p = worldXY(e);
  var x = Math.round(p.x), y = Math.round(p.y);
  if (x < 0 || y < 0 || x >= LOG || y >= LOG) return;
  var rec = loadedViews[curView];
  var vid = rec.vid[y * LOG + x];
  if (vid >= MANIFEST.nvert) return;
  setMark(LAND[cur][0], curView, vid);
}
function setMark(key, view, vid) {
  if (!marks[key]) marks[key] = {};
  marks[key][view] = { vertex: vid };
  undoStack.push({ k: key, v: view }); if (undoStack.length > 300) undoStack.shift();
  save();
  for (var tgt in MSRC) {
    if (MSRC[tgt] === key && !marks[tgt] && view !== "side_left" && view !== "side_right") {
      var mv = MIRROR[vid];
      if (mv < MANIFEST.nvert) setMark(tgt, view, mv);
    }
  }
  if (cur < LAND.length - 1) cur++;
  refreshList(); drawView();
}
function save() { try { localStorage.setItem(STORE, JSON.stringify(marks)); } catch (e) {} }
function refreshList() {
  var box = $("lmList"); box.innerHTML = "";
  LAND.forEach(function (lm, i) {
    var d = document.createElement("div");
    var has = !!marks[lm[0]];
    d.textContent = (has ? "✓ " : "○ ") + i + " " + lm[1];
    d.className = (i === cur ? "sel " : "") + (has ? "done" : "");
    d.onclick = function () { cur = i; refreshList(); drawView(); };
    box.appendChild(d);
  });
  var sel = LAND[cur];
  $("prog").textContent = (cur + 1) + "/" + LAND.length + " · " + sel[1];
  $("info").innerHTML = "<b>" + sel[0] + "（" + sel[1] + "）</b>" + sel[2] +
    "<div style='color:#667;margin-top:4px'>主视图: " + sel[4] + (sel[3] ? " · 自动镜像 → " + sel[3] : "") + "</div>";
  $("vstat").textContent = "完成 " + countDone() + "/" + LAND.length;
}
function countDone() { var n = 0; LAND.forEach(function (lm) { if (marks[lm[0]]) n++; }); return n; }
function onKey(e) {
  var t = e.target;
  if (t && (t.tagName === "TEXTAREA" || t.tagName === "INPUT")) return;
  if (e.key === "ArrowRight") UI.nextL();
  else if (e.key === "ArrowLeft") UI.prevL();
  else if (e.key === "z" || e.key === "Z") UI.undo();
  else if (e.key === " ") { e.preventDefault(); UI.nextL(); }
}
var UI = {
  nextL: function () { cur = (cur + 1) % LAND.length; refreshList(); drawView(); },
  prevL: function () { cur = (cur + LAND.length - 1) % LAND.length; refreshList(); drawView(); },
  undo: function () {
    var t = undoStack.pop(); if (!t) return;
    if (marks[t.k] && marks[t.k][t.v]) delete marks[t.k][t.v];
    if (marks[t.k] && !Object.keys(marks[t.k]).length) delete marks[t.k];
    save(); refreshList(); drawView();
  },
  exportJ: function () {
    var miss = LAND.filter(function (lm) { return !marks[lm[0]]; }).map(function (lm) { return lm[1]; });
    if (miss.length && !confirm("还有 " + miss.length + " 个未标:\n" + miss.join("、") + "\n\n仍要导出?")) return;
    var out = {};
    LAND.forEach(function (lm) {
      var src = marks[lm[0]]; if (!src) return;
      var sx = 0, sy = 0, sz = 0, n = 0, per = {};
      for (var v in src) {
        var vid = src[v].vertex;
        if (vid === undefined || vid >= MANIFEST.nvert) continue;
        var o = vid * 3;
        sx += VERTS[o]; sy += VERTS[o + 1]; sz += VERTS[o + 2]; n++;
        per[v] = { vertex: vid };
      }
      if (!n) return;
      var mx = sx / n, my = sy / n, mz = sz / n, best = 0, bd = 1e18;
      for (var i = 0; i < MANIFEST.nvert; i++) {
        var dx = VERTS[i * 3] - mx, dy = VERTS[i * 3 + 1] - my, dz = VERTS[i * 3 + 2] - mz;
        var dd = dx * dx + dy * dy + dz * dz;
        if (dd < bd) { bd = dd; best = i; }
      }
      out[lm[0]] = { pos: [+mx.toFixed(3), +my.toFixed(3), +mz.toFixed(3)], vertex: best, sources: per };
    });
    var txt = JSON.stringify(out, null, 1);
    var o = $("out"); o.style.display = "block"; o.textContent = txt;
    try { navigator.clipboard.writeText(txt); } catch (e) {}
    try {
      var b = new Blob([txt], { type: "application/json" });
      var a = document.createElement("a");
      a.href = URL.createObjectURL(b); a.download = "rocky_landmark_vertex_map.json"; a.click();
    } catch (e) {}
  }
};
boot();
