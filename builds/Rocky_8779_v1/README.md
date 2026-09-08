# Rocky 8779 v1 — geometry candidate

状态：`GEOMETRY_READY_TEXTURE_BLOCKED`

已冻结：

- native 8779 container topology: 24,484 vertices / 78,412 faces
- geometry source: existing `rocky_target_v02_head.obj` child-proportion pass
- native face index order preserved
- no new 3DMM, landmark system, camera fit, NRICP, or photogrammetry

尚未完成：

- `face_color`: native IFF / UV buffer / DDS source is not present in this Mac workspace
- native normal/wrinkle source: not present as a writable game asset
- compatible short-hair asset: not present locally
- Windows IFF packaging and NBA 2K26 test

因此该目录是可继续打包的 v1 geometry candidate，不宣称已经完成可安装的
游戏成品。需要 native 8779 IFF/UV/texture/hair assets before the Windows test.
