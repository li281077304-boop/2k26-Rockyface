# NBA 2K26 真人照片 → 自定义 Cyberface 制作工具链

**项目**：李谨一（Rocky）/ Vít Krejčí @ Face ID 8779（沙箱项目；李凡/杨瀚森不在此库）
**仓库属性**：Private。**照片不入库**（经 iCloud / 局域网 / U 盘传输）。

## 分工

| 机器 | 职责 |
|---|---|
| **Mac**（建议） | 3DDFA_V2 ONNX → 3D 真人头 → Procrustes → trimesh `nricp_amberg/nricp_sumner` → 8779 拓扑迁移 → OBJ 预览 → 贴图生成 |
| **Windows** | 8779 IFF 重新打包、mods 安装、DB2K / Face ID、MyNBA 存档、游戏内验收 |

## 主线管线

```
谨一照片(多视角) → 3DDFA_V2(dense 3DMM) → canonical Rocky head
        → similarity align → native 8779 head
        → nricp_amberg / nricp_sumner (保留 8779 拓扑/顶点序/UV/skin weights)
        → rocky_8779_nricp_v0x.obj → (回 Windows) IFF 打包 → 游戏内验收
```

已破解的 8779 IFF 结构见 `configs/8779_FORMAT.md`；各轮状态见 `reports/STATUS_20260908.md`。

## Mac 起步（不装 torch，先用官方 ONNX 路线）

```bash
brew install python@3.11 libomp git
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy scipy opencv-python pyyaml matplotlib trimesh onnxruntime
# 3DDFA_V2 需先有 3 个 .onnx：FaceBoxesProd / mb1_120x120 / bfm_noneck_v3
# （可先在一台有 torch 的机器导一次，或取得预导出文件）—— 见 docs/ONNX_SETUP.md
python demo.py -f <正脸.jpg> -o obj --onnx   # → 得到 rocky_3ddfa_front.obj
```

**极小目标**：谨一正脸 → `rocky_3ddfa_front.obj` 半小时内跑通，即证明换机方向正确。

## Windows 侧（本次会话已产出）

- 8779 全身 IFF 解析：head=Y≥55，24,484 顶点；坐标 float32×3；索引 uint16（78,412 三角形）
- `hihead.SCNE` 为 **JSON 明文**；UV 位于 `VertexBuffer.7c6afba5…` stride16/offset4(uint16 UNORM)
- `scripts/` 可复用工具；`annotator_desktop/` 是目录式(localhost)解剖点标定器（可选辅助）

## 目录

```
scripts/    解析/渲染/形变/标注器生成脚本
docs/       任务书、ONNX_SETUP、重建结论
configs/    8779 格式白皮书
geometry/   native 8779 OBJ 与形变候选
reports/    分轮状态报告
annotator_desktop/  Windows 用的 25 点标定工具（浏览器 + 后端 http.server）
```

## 禁止入库

照片、.venv、*.iff、*.dds、backups、tmp、临时渲染、几十 MB 备份、任何游戏目录内容。
大文件（OBJ 超过 ~50MB 场景）用 Git LFS。
