# 3DDFA_V2 ONNX 路线 — 阻塞与解锁说明（2026-09-08 实测）

## 阻塞结论（本机已验证）

1. 仓库内**没有** `.onnx` 权重：`find . -iname "*.onnx"` 为空
2. GitHub Releases（v0.1/v0.11/v0.12）**没有任何 asset**
3. 仓库自带导出脚本 **强制 import torch**：
   - `utils/onnx.py`（TDDFA → mb1_120x120.onnx）
   - `FaceBoxes/onnx.py`（→ FaceBoxesProd.onnx）
   - `bfm/bfm_onnx.py`（→ bfm_noneck_v3.onnx）
   且 `TDDFA_ONNX.py` 顶部 `from utils.onnx import convert_to_onnx` 会让
   **无 torch 环境下 import TDDFA_ONNX 直接失败**（utils.onnx 顶层 import torch）
4. 唯一其它下载途径是 Google Drive（本机直连 502）

→ 结论：**纯 onnxruntime 无 torch 的路线，在本仓库结构下无法启动**，
缺口正是这三个 onnx 文件，不在"装库"而在"权重资产"。

## 需要的三个文件（放到指定路径即解锁）

| 目标路径 | 来源 |
|---|---|
| `weights/FaceBoxesProd.onnx` | FaceBoxes 导出 |
| `weights/mb1_120x120.onnx` | TDDFA 主干导出 |
| `configs/bfm_noneck_v3.onnx` | BFM 参数导出 |

## 解锁方式（任选其一）

### A. 任何有 torch 的机器上导出一次（约 5 行，本机以外的环境）
```python
import sys; sys.path.insert(0, "3DDFA_V2")
from utils.onnx import convert_to_onnx          # 需要 torch/torchvision
from FaceBoxes.onnx import convert_to_onnx as fconvert
from bfm.bfm_onnx import convert_bfm_to_onnx
import yaml, os
cfg = yaml.safe_load(open("3DDFA_V2/configs/mb1_120x120.yml"))
convert_to_onnx(**cfg)          # -> weights/mb1_120x120.onnx
convert_bfm_to_onnx("3DDFA_V2/configs/bfm_noneck_v3.onnx",
                    "3DDFA_V2/configs/bfm_noneck_v3.pkl", img_size=120)
fconvert("3DDFA_V2/weights/FaceBoxesProd.onnx")
```
（更省事：把 `H:\2k面补\3DDFA_V2` 拷到该机器直接跑上面几行，再把三个 .onnx 拷回）

### B. 从可信来源获取已导出的三份 .onnx 放回上述路径

## 本机已就绪（权重一到就能跑）

- Rocky venv 已装 `onnxruntime 1.29.0`（numpy/opencv/pyyaml 齐）
- 推理环境 python：
  `H:\2k面补\Rocky_8779_rebuild\.venv\Scripts\python.exe`
- 待权重就位后运行：
  `python demo.py -f inputs_rocky/r01_front.jpg -o obj --onnx`
  （或最小脚本只调 FaceBoxes_ONNX + TDDFA_ONNX + recon_vers + ser_to_obj）
