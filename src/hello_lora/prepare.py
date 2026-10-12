"""阶段二：把「图片 + danbooru 风格 metadata.jsonl」转换成 kohya 训练结构。

产物全部归属数据集目录自身，保持数据集自包含:

    <dataset_dir>/
    ├─ *.jpg / *.png / *.webp     # 原始图片（只读，不修改）
    ├─ metadata.jsonl             # danbooru 风格元数据（只读）
    ├─ train/<repeats>_<concept>/ # kohya 结构: 图片(硬链接) + 同名 .txt caption
    ├─ val/                       # --val-ratio > 0 时的验证集副本（kohya 看不到）
    └─ manifest/prepare.parquet

caption 来源按优先级取其一：数据集目录下已有的 <stem>.txt（WD14 打标产物，
为无元数据的数据集如 Kaggle 提供来源）> metadata.jsonl 构建。
后者默认取 tag_string_general + tag_string_copyright，去掉 artist 和 meta 标签
（画风训练不应绑定画师名，meta 标签是画质标记）。
kohya 的 caption 是纯文本，不做 WebUI 式括号转义。

train/val 切分由 --val-ratio 控制：按 image_id（内容哈希）确定性划分，
同一图片无论重跑多少次、扩库后归属都稳定；val 图不进训练目录，
供评估阶段做泛化检测（hello-lora eval）。不传 --val-ratio 时行为与
旧版完全一致（全部 train）。

manifest 的 image_id 与 scan 阶段口径一致（sha256 前 16 位），
两个阶段的 manifest 可直接按 image_id 对齐；txt-only 的图片缺少
booru 元数据，对应列为 null。
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import polars as pl
from PIL import Image

from hello_lora.scan import sha256_file

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

PREPARE_SCHEMA = {
    "image_id": pl.String,  # sha256 前 16 位，与 scan 口径一致
    "path": pl.String,  # 训练目录中的绝对路径（kohya 需要）
    "md5": pl.String,  # booru 帖子 md5（即文件名 stem）
    "width": pl.Int64,
    "height": pl.Int64,
    "aspect_ratio": pl.Float64,
    "file_size": pl.Int64,
    "rating": pl.String,
    "score": pl.Int64,
    "tag_count_general": pl.Int64,
    "split": pl.String,
    "notes": pl.String,
}


def build_caption(record: dict, trigger_word: str | None, keep_artist: bool) -> str:
    tags = record.get("tag_string_general", "").split()
    tags += record.get("tag_string_copyright", "").split()
    if keep_artist:
        tags += record.get("tag_string_artist", "").split()
    caption = ", ".join(tag.replace("_", " ") for tag in tags)
    if trigger_word:
        caption = f"{trigger_word}, {caption}"
    return caption


def assign_split(image_id: str, val_ratio: float) -> str:
    """按 image_id（内容哈希前 16 位）确定性划分 train/val。

    哈希取模保证同一图片重跑、扩库后归属稳定，与文件顺序无关。
    """
    if val_ratio <= 0:
        return "train"
    return "val" if int(image_id, 16) % 10_000 < round(val_ratio * 10_000) else "train"


def prepare_dataset(
    dataset_dir: Path,
    repeats: int = 1,
    ratings: list[str] | None = None,
    trigger_word: str | None = None,
    keep_artist_tag: bool = False,
    val_ratio: float = 0.0,
) -> tuple[pl.DataFrame, Path, list[tuple[str, str]]]:
    """准备数据集，返回 (manifest, 训练目录, 跳过列表)。

    跳过列表元素为 (文件名, 原因)，覆盖无元数据、评分过滤和解码失败。
    val_ratio > 0 时按 image_id 确定性切出验证集，硬链接到 <dataset>/val/，
    不进 kohya 训练目录，供评估阶段做泛化检测。
    """
    if not dataset_dir.is_dir():
        raise SystemExit(f"数据集目录不存在: {dataset_dir}")
    if not 0 <= val_ratio < 1:
        raise SystemExit(f"val_ratio 需在 [0, 1) 内: {val_ratio}")
    metadata_path = dataset_dir / "metadata.jsonl"
    records: dict[str, dict] = {}
    if metadata_path.is_file():
        records = {
            r["md5"]: r
            for r in (
                json.loads(line)
                for line in metadata_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }

    images = sorted(p for p in dataset_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    # caption 来源二选一：metadata.jsonl 或打标产物 <stem>.txt
    has_tag_txt = any((dataset_dir / f"{img.stem}.txt").is_file() for img in images)
    if not records and not has_tag_txt:
        raise SystemExit(
            f"缺少 caption 来源: {dataset_dir} 下既无 metadata.jsonl 也无 <stem>.txt\n"
            "（先运行 hello-lora tag 生成标签，或提供 danbooru 风格 metadata.jsonl）"
        )

    # 概念目录名: 数据集目录名净化为 kohya 可用的 <repeats>_<concept> 形式
    concept = re.sub(r"[^0-9A-Za-z]+", "_", dataset_dir.name).strip("_").lower()
    train_dir = dataset_dir / "train" / f"{repeats}_{concept}"
    manifest_dir = dataset_dir / "manifest"
    val_dir = dataset_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    if val_ratio > 0:
        val_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped: list[tuple[str, str]] = []
    for img in images:
        record = records.get(img.stem)
        source_txt = dataset_dir / f"{img.stem}.txt"
        if record is None and not source_txt.is_file():
            skipped.append((img.name, "无元数据"))
            continue
        if record is None and ratings:
            # 评分过滤无法作用于 txt-only 的图片，直接视为不通过
            skipped.append((img.name, "无元数据, 无法按评分过滤"))
            continue
        if record is not None and ratings and record.get("rating") not in ratings:
            skipped.append((img.name, f"评分过滤: {record.get('rating')}"))
            continue
        # 完整解码校验，剔除损坏文件
        try:
            with Image.open(img) as im:
                im.verify()
        except Exception as exc:
            skipped.append((img.name, f"解码失败: {exc}"))
            continue

        # 确定性切分：由内容哈希决定 train/val，与文件顺序和重跑次数无关
        image_id = sha256_file(img)[:16]
        split = assign_split(image_id, val_ratio)
        target_dir = val_dir if split == "val" else train_dir

        # 训练/验证目录里的图片用硬链接引用根目录原件（同一卷上不占额外空间）；
        # 文件系统不支持硬链接（exFAT、跨卷）时回退为完整复制
        linked = target_dir / img.name
        linked.unlink(missing_ok=True)
        try:
            os.link(img, linked)
        except OSError:
            shutil.copy2(img, linked)
        # caption 优先用打标产物；有触发词时同样前缀
        if source_txt.is_file():
            caption = source_txt.read_text(encoding="utf-8").strip()
            if trigger_word:
                caption = f"{trigger_word}, {caption}"
        else:
            caption = build_caption(record, trigger_word, keep_artist_tag)
        (target_dir / f"{img.stem}.txt").write_text(caption, encoding="utf-8")

        if record is not None:
            width, height = record["image_width"], record["image_height"]
            file_size = record["file_size"]
        else:
            with Image.open(img) as im:
                width, height = im.size
            file_size = img.stat().st_size
        rows.append(
            {
                "image_id": image_id,
                "path": str(target_dir / img.name),
                "md5": record["md5"] if record else None,
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 4),
                "file_size": file_size,
                "rating": record.get("rating") if record else None,
                "score": record.get("score") if record else None,
                "tag_count_general": record.get("tag_count_general") if record else None,
                "split": split,
                "notes": "",
            }
        )

    # 显式 schema：全部被跳过时 rows 为空也能建出带列名的空表
    manifest = pl.DataFrame(rows, schema=PREPARE_SCHEMA)
    manifest_path = manifest_dir / "prepare.parquet"
    manifest.write_parquet(manifest_path)
    _backfill_split(manifest_dir, manifest)
    return manifest, train_dir, skipped


def _backfill_split(manifest_dir: Path, manifest: pl.DataFrame) -> None:
    """scan 清单存在时，把 split 列回填进去（切分步骤兑现 scan 预留的列）。"""
    scan_path = manifest_dir / "scan.parquet"
    if not scan_path.is_file() or manifest.height == 0:
        return
    scan_manifest = pl.read_parquet(scan_path)
    if "split" not in scan_manifest.columns:
        return
    updates = manifest.select(
        pl.col("image_id"), pl.col("split").alias("split_from_prepare")
    )
    scan_manifest = (
        scan_manifest.join(updates, on="image_id", how="left")
        .with_columns(
            pl.coalesce("split_from_prepare", "split").alias("split")
        )
        .drop("split_from_prepare")
        .sort("path")
    )
    scan_manifest.write_parquet(scan_path)


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
