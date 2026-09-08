# NBA 2K26 8779 IFF 格式白皮书（已实测破解，2026-09-08）

基于 `D:\STEAM\steamapps\common\NBA 2K26` 原生 `RobbieAvila8779\png8779.iff`。

## 1. IFF = ZIP

IFF 是普通 zip；每个 entry 是资源文件。`png8779.iff` 主档约 375 条目（原生），
**当前游戏 mods 中那份伪 8779 是 4133 整包改名（661 条目），作废勿用。**

## 2. hihead.SCNE = JSON 明文（关键）

- 顶网格顶点数：`"LodVerts": 24484`（LOD0）；还有 LOD1 等
- 包围盒：`Min [-78.05,-103.63,-12.88]  Max [78.05,81.14,27.47]`（X 左右 / Y 上下 / Z 前后，脸在 +Z）
- `"VertexStream": [...]` 数组声明每个流的 Stride/Size/Binary
- Morph 为**稀疏 R8_UINT 增量**，`Scale: 0.015625`（1/64），SCNE 内按名声明（如 `brow_innerDown_L`）

## 3. 顶点/索引缓冲

| 资源 | 结构 |
|---|---|
| 坐标 | `VertexBuffer.f2fce607cb4c3b42.bin`，24484 × float32[3]，Stride 12 |
| 索引 | `IndexBuffer.c829179a1bc03cf2.bin`，uint16，235,236 个（78,412 三角形），max=24483 |
| **UV** | `VertexBuffer.7c6afba5d91ba744.bin`，**Stride 16，字节偏移 4–7，uint16 UNORM /65535** |
| 顶点流2前段 | 同上缓冲 offset0–3/8–15：打包法线/切线 |
| 权重 | `MatrixWeightsBuffer.313dc1adc1fb35a9.bin`，157,960 B |
| 顶点流4 | `VertexBuffer.1ba830a661e6c481.bin`，Stride 4，1,119,336 B（非 24484 整除，用途待定）|

面部位 UV 子区约：U 0.40–0.60 / V 0.05–0.30（中央偏上椭圆）。
整份 UV 展开其实是**全身角色**布局（头=Y≥55 约 11,560 顶点 / 31,810 三角形）。

## 4. 验证工具（scripts/）

- `find_uv.py`：对 (buffer,offset,encoding) 穷举，以 UV/3D 边长对数相关性为杀手判据 → 命中上表
- `meshkit.py`：OBJ 读写 + 软件光栅化（含 pick 缓冲：每像素 tri id + 重心坐标）
- `make_vertex_map.py` / `build_annotator*.py` / `build_site.py`：头模解剖点标定（浏览器）
- `fit_rocky_v04.py`：⚠️ **EXPERIMENTAL/COARSE INIT**，仅初始化用（多照片 PnP 三角化），非主线

## 5. 重打包注意

- DDS 为 BC7（DX10 fourcc），mip 12 级；texconv 2026 版格式名须写 `BC7_UNORM`
- 修好 texture 后 repack IFF：保持 zip 条目顺序/压缩类型；CRC 校验通过≠游戏可加载
- 改头模须同步 SCNE 的 Min/Max/Center/Radius
