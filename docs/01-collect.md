# 1. 图片采集

从 booru 站点或 Kaggle 把图片拉进 `assets/data/<dataset>/`，一个数据集一个目录。

```bash
# Danbooru 按艺术家采集（按标签用 --tags "<tag1>,<tag2>"）
uv run python scripts/danbooru.py --artist "<artist>" --limit 200

# Konachan 按标签采集（Moebooru API；默认走 konachan.net 镜像，
# --sample 下载 1500px 压缩版，--rating safe|questionable|explicit 按分级过滤）
uv run python scripts/konachan.py --tags "<tag1>,<tag2>" --limit 200

# Kaggle 数据集采集
uv run python scripts/kaggle.py
```

## 共同行为

- 文件以帖子 md5 命名（内容寻址，天然防重复下载）；
- 帖子元数据追加写入 `<out>/metadata.jsonl`（danbooru 风格，后续 prepare 用）；
- 断点续传：已存在的文件跳过并做 md5 内容校验；
- 代理：默认取运行环境 profile 的 `[network].proxy`（没有则直连），
  `--proxy <地址>` 覆盖，`--proxy ""` 强制直连；
- `--username/--api-key` 配 Danbooru 账号（更高速率上限）。
