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

## Mac Phase 1 — 3DDFA（先做这个）

第一验收物：

```
geometry/rocky_3ddfa_front.obj
```

**在这个文件出来以前，不执行 NRICP。** 前置：3 个 ONNX 权重就位（见 `docs/ONNX_SETUP.md`），
照片经 iCloud/局域网传入（不入库）。

## Mac Phase 2 — canonical

多张照片 3DDFA 全部成功后：

1. Procrustes / similarity align
2. 排除明显异常结果（表情/侧脸不稳/重建失败）
3. 融合 identity shape

输出：

```
geometry/rocky_3ddfa_canonical.obj
```

## Mac Phase 3 — transfer to native 8779

**重要方向（易错）：**

```
source = native 8779（保持不变）
target = Rocky canonical（形状目标）
native 8779 → deform toward Rocky source → output 仍是 8779 topology
```

原因：最终必须保留 8779 的 vertex count / vertex order / topology / UV / skin weights。
**不要覆盖 `geometry/native_8779.obj`。**

使用（先验证 API 可 import）：

```python
import trimesh
from trimesh.registration import nricp_amberg, nricp_sumner
```

分别输出：

```
geometry/rocky_8779_amberg_v01.obj
geometry/rocky_8779_sumner_v01.obj
```

NRICP 前必须先 similarity / rigid align；必要时仅用少量 semantic landmarks 做初始化。
**RBF 不是最终 dense transfer**，只作为 initialization / landmark refinement / local correction。

如果两者都成功，比较：face length / cheek volume / nose bridge / nose tip / eye socket /
jaw / chin / surface smoothness / 是否出现塌陷折叠。

## Mac Phase 4 — preview only

输出五视角：front / left45 / right45 / left90 / right90。

在用户确认几何以前**禁止**：写回 IFF、texture、Morph、hair、DB2K、改 Face ID。

## Windows handoff

Mac 最终只需回传：

```
rocky_8779_*_best.obj
+ fitting report
+ preview images
```

Windows 再负责：OBJ/vertex data → 8779 VertexBuffer → native 8779 IFF → mods → 游戏内验收。

## 写死的当前结论（勿再推翻）

- **4133 = 杨瀚森（310s 原版），禁止修改**
- **8779 = Rocky（李谨一）**；Face ID 已定 8779，不再换槽位
- **native 8779 container 必须保留**为唯一合法基底
- **旧「4133 整包改名 8779」的构建已废弃**（661 条目伪 8779 作废）
- 照片永不入库；`.gitignore` 已锁死 *.jpg/*.png/*.iff/*.dds/*.bin/*.npy

## 禁止入库

照片、.venv、*.iff、*.dds、backups、tmp、临时渲染、几十 MB 备份、任何游戏目录内容。
大文件（OBJ 超过 ~50MB 场景）用 Git LFS。
