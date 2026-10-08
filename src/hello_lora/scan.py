"""阶段二：扫描图片目录，生成 image_manifest.parquet 和扫描报告。

只读取输入目录，不修改任何图片；清洗结果由后续步骤写入 data/cleaned/。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
from PIL import Image, UnidentifiedImageError

# 质量阈值：触发任一条件的图片进入待人工确认清单
MIN_SIDE = 512  # 最小边长，低于此值不利于 SDXL 训练
EXTREME_ASPECT = 3.0  # 宽高比超过此值视为极端比例

# 全项目接受的图片格式（大写，取自 PIL Image.format）
SUPPORTED_FORMATS = {"PNG", "JPEG", "WEBP"}

MANIFEST_SCHEMA = {
    "image_id": pl.String,  # sha256 前 16 位，内容寻址，天然标识完全重复
    "path": pl.String,  # 相对输入目录的路径
    "width": pl.Int32,
    "height": pl.Int32,
    "aspect_ratio": pl.Float32,
    "format": pl.String,  # 实际解码出的格式，而非扩展名
    "file_size": pl.Int64,
    "sha256": pl.String,
    "perceptual_hash": pl.String,  # 由去重步骤回填
    "duplicate_group": pl.String,  # 由去重步骤回填
    "quality_status": pl.String,  # ok / corrupt / unsupported
    "caption_status": pl.String,  # 由标注步骤回填
    "split": pl.String,  # 由数据集切分步骤回填
    "notes": pl.String,  # 质量标记，分号分隔；非空即待人工确认
}


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def scan_single_file(path: Path) -> dict:
    """扫描单个文件，返回一行 manifest 记录。损坏文件也会被记录而非跳过。"""
    record = {name: None for name in MANIFEST_SCHEMA}
    record.update(
        file_size=path.stat().st_size,
        quality_status="unsupported",
        notes="",
    )
    record["sha256"] = sha256_file(path)
    record["image_id"] = record["sha256"][:16]

    try:
        with Image.open(path) as img:
            img.verify()  # 校验文件完整性；verify 后需要重新打开读取属性
        with Image.open(path) as img:
            width, height = img.size
            pil_format = img.format
    except (UnidentifiedImageError, OSError) as exc:
        # 扩展名像图片但解码失败 -> corrupt（需人工确认）；否则视为无关文件 -> unsupported
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            record["quality_status"] = "corrupt"
            record["notes"] = f"decode_failed: {exc.__class__.__name__}"
        return record

    aspect = width / height
    notes = []
    if pil_format not in SUPPORTED_FORMATS:
        record["quality_status"] = "unsupported"
        record["notes"] = f"format_not_accepted: {pil_format}"
        return record

    record.update(
        quality_status="ok",
        width=width,
        height=height,
        aspect_ratio=aspect,
        format=pil_format,
    )
    if min(width, height) < MIN_SIDE:
        notes.append(f"too_small(<{MIN_SIDE})")
    if max(aspect, 1 / aspect) > EXTREME_ASPECT:
        notes.append(f"extreme_aspect(>{EXTREME_ASPECT})")
    record["notes"] = ";".join(notes)
    return record


def scan_directory(input_dir: Path) -> pl.DataFrame:
    """递归扫描目录下所有文件，生成符合 MANIFEST_SCHEMA 的清单。

    path 记录相对输入目录的路径，保证清单与目录移动无关。
    """
    files = sorted(p for p in input_dir.rglob("*") if p.is_file())
    records = [scan_single_file(p) for p in files]
    for record, path in zip(records, files):
        record["path"] = path.relative_to(input_dir).as_posix()
    return pl.DataFrame(records, schema=MANIFEST_SCHEMA)


def summarize(manifest: pl.DataFrame) -> dict:
    """生成扫描报告：总数、按格式/状态聚合、尺寸统计、待人工确认清单。"""
    ok = manifest.filter(pl.col("quality_status") == "ok")
    report = {
        "total_files": manifest.height,
        "by_quality_status": manifest.group_by("quality_status").len().sort("quality_status").to_dicts(),
        "by_format": ok.group_by("format").len().sort("format").to_dicts(),
        "size_stats": ok.select(
            pl.col("width").min().alias("width_min"),
            pl.col("width").max().alias("width_max"),
            pl.col("height").min().alias("height_min"),
            pl.col("height").max().alias("height_max"),
            pl.col("aspect_ratio").min().alias("aspect_min"),
            pl.col("aspect_ratio").max().alias("aspect_max"),
        ).to_dicts()[0],
        "needs_review": manifest.filter(
            (pl.col("notes").is_not_null()) & (pl.col("notes") != "")
        )["path"].to_list(),
        "duplicate_exact_groups": ok.group_by("image_id").len().filter(pl.col("len") > 1).height,
    }
    return report
