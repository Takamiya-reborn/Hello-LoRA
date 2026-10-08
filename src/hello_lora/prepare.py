"""阶段二：把「图片 + danbooru 风格 metadata.jsonl」转换成 kohya 训练结构。

产物全部归属数据集目录自身，保持数据集自包含:

    <dataset_dir>/
    ├─ *.jpg / *.png / *.webp     # 原始图片（只读，不修改）
    ├─ metadata.jsonl             # danbooru 风格元数据（只读）
    ├─ train/<repeats>_<concept>/ # kohya 结构: 图片 + 同名 .txt caption
    └─ manifest/image_manifest.parquet

caption 默认取 tag_string_general + tag_string_copyright，
去掉 artist 和 meta 标签（画风训练不应绑定画师名，meta 标签是画质标记）。
kohya 的 caption 是纯文本，不做 WebUI 式括号转义。
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import polars as pl
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def build_caption(record: dict, trigger_word: str | None, keep_artist: bool) -> str:
    tags = record.get("tag_string_general", "").split()
    tags += record.get("tag_string_copyright", "").split()
    if keep_artist:
        tags += record.get("tag_string_artist", "").split()
    caption = ", ".join(tag.replace("_", " ") for tag in tags)
    if trigger_word:
        caption = f"{trigger_word}, {caption}"
    return caption


def prepare_dataset(
    dataset_dir: Path,
    repeats: int = 1,
    ratings: list[str] | None = None,
    trigger_word: str | None = None,
    keep_artist_tag: bool = False,
) -> tuple[pl.DataFrame, Path, list[tuple[str, str]]]:
    """准备数据集，返回 (manifest, 训练目录, 跳过列表)。

    跳过列表元素为 (文件名, 原因)，覆盖无元数据、评分过滤和解码失败。
    """
    if not dataset_dir.is_dir():
        raise SystemExit(f"数据集目录不存在: {dataset_dir}")
    metadata_path = dataset_dir / "metadata.jsonl"
    if not metadata_path.is_file():
        raise SystemExit(f"缺少 metadata.jsonl: {metadata_path}")

    # 概念目录名: 数据集目录名净化为 kohya 可用的 <repeats>_<concept> 形式
    concept = re.sub(r"[^0-9A-Za-z]+", "_", dataset_dir.name).strip("_").lower()
    train_dir = dataset_dir / "train" / f"{repeats}_{concept}"
    manifest_dir = dataset_dir / "manifest"
    train_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    records = {
        r["md5"]: r
        for r in (
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }

    images = sorted(p for p in dataset_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    rows = []
    skipped: list[tuple[str, str]] = []
    for img in images:
        record = records.get(img.stem)
        if record is None:
            skipped.append((img.name, "无元数据"))
            continue
        if ratings and record.get("rating") not in ratings:
            skipped.append((img.name, f"评分过滤: {record.get('rating')}"))
            continue
        # 完整解码校验，剔除损坏文件
        try:
            with Image.open(img) as im:
                im.verify()
        except Exception as exc:
            skipped.append((img.name, f"解码失败: {exc}"))
            continue

        shutil.copy2(img, train_dir / img.name)
        caption = build_caption(record, trigger_word, keep_artist_tag)
        (train_dir / f"{img.stem}.txt").write_text(caption, encoding="utf-8")
        rows.append(
            {
                "image_id": record["id"],
                "path": str(train_dir / img.name),
                "md5": record["md5"],
                "width": record["image_width"],
                "height": record["image_height"],
                "aspect_ratio": round(
                    record["image_width"] / record["image_height"], 4
                ),
                "file_size": record["file_size"],
                "rating": record["rating"],
                "score": record["score"],
                "tag_count_general": record["tag_count_general"],
                "split": "train",
                "notes": "",
            }
        )

    manifest = pl.DataFrame(rows)
    manifest_path = manifest_dir / "image_manifest.parquet"
    manifest.write_parquet(manifest_path)
    return manifest, train_dir, skipped


def summarize_dataset(manifest: pl.DataFrame) -> str:
    """生成训练用 manifest 的人读摘要。"""
    n, min_w, min_h, min_ar, max_ar = manifest.select(
        pl.len(),
        pl.col("width").min(),
        pl.col("height").min(),
        pl.col("aspect_ratio").min().alias("min_ar"),
        pl.col("aspect_ratio").max().alias("max_ar"),
    ).row(0)
    return f"尺寸概况: n={n}, min_w={min_w}, min_h={min_h}, ar=[{min_ar}, {max_ar}]"
