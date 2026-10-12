# 2. 清洗（扫描）

```bash
uv run hello-lora scan <dataset>
#    可选: --no-cache 强制全量重扫（默认增量）
```

生成 manifest 清单（`<dataset>/manifest/scan.parquet`）+ 扫描报告
（`<dataset>/manifest/scan.report.json`）。

- 只扫描图片文件（jpg / jpeg / png / webp），自动跳过 `metadata.jsonl` 和
  `manifest/`、`train/`、`val/` 等流水线保留目录；
- 每张图记录尺寸、格式、sha256（`image_id` = 前 16 位）、感知哈希 phash；
- 质量标记：最小边长 < 512、宽高比 > 3.0 的图进待人工确认清单（`needs_review`）；
- **增量执行**：按 file_size + mtime 复用上次结果，新增或修改过的文件才重扫；
  `--no-cache` 强制全量重扫；
- `<dataset>` 填 `assets/data/` 下的目录名即可，传已存在的目录路径则原样使用。
