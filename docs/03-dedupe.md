# 3. 去重

由上一步 `scan` 的报告（`<dataset>/manifest/scan.report.json`）覆盖，待人工确认后删除：

- **完全重复**（sha256 相同）：组数见 `duplicate_exact_groups`，
  成员路径见 `duplicate_exact_paths`；
- **近重复**（感知哈希 phash 汉明距离 ≤ 4，如同图不同编码 / 轻微裁剪）：
  分组路径见 `duplicate_near_groups`，每组保留一张删其余。

近重复分组用并查集求连通分量：组内任意两张不一定直接相似，
但经由传递闭包相连（A 像 B、B 像 C，则 A/B/C 同组）。
