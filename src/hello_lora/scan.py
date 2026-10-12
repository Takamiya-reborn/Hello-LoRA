"""阶段二：扫描图片目录，生成 scan.parquet 和扫描报告。

只读取输入目录，不修改任何图片。清洗范围约定：

- 只扫描图片扩展名的文件，非图片文件不进 manifest；
- 数据集目录中的保留名（manifest / train / val）一律跳过，
  避免把流水线产物和 prepare 的训练/验证副本扫成重复；
- metadata.jsonl 因扩展名不在 IMAGE_EXTS 内天然被排除。
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
import polars as pl
from PIL import Image, UnidentifiedImageError

# 质量阈值：触发任一条件的图片进入待人工确认清单
MIN_SIDE = 512  # 最小边长，低于此值不利于 SDXL 训练
EXTREME_ASPECT = 3.0  # 宽高比超过此值视为极端比例
NEAR_DUP_DISTANCE = 4  # phash 汉明距离 <= 此值视为近重复
MAX_WORKERS = 16  # 并行扫描线程数（解码和哈希多为释放 GIL 的操作）

# 全项目接受的图片格式（大写，取自 PIL Image.format）
SUPPORTED_FORMATS = {"PNG", "JPEG", "WEBP"}
# 只扫描这些扩展名，非图片文件不进 manifest
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
# 数据集目录中的保留名：这些名字出现在相对路径的任意父目录层级即跳过。
# train / val 是 prepare 的训练、验证副本（--val-ratio 切分）；
# 训练实验产物在 assets/experiments/（与数据集平级），不在数据集目录内
RESERVED_DIRS = {"manifest", "train", "val"}

MANIFEST_SCHEMA = {
    "image_id": pl.String,  # sha256 前 16 位，内容寻址，天然标识完全重复
    "path": pl.String,  # 相对输入目录的路径
    "width": pl.Int32,
    "height": pl.Int32,
    "aspect_ratio": pl.Float32,  # 宽/高
    "format": pl.String,  # 实际解码出的格式，而非扩展名
    "file_size": pl.Int64,
    "mtime": pl.Float64,  # 修改时间，与 file_size 一起用于增量重扫
    "sha256": pl.String,  # 内容哈希
    "perceptual_hash": pl.String,  # phash 十六进制串，近重复去重依据
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
        mtime=path.stat().st_mtime,
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
            # phash 需要像素数据，顺带完成全解码，可检出 verify 漏掉的截断文件
            record["perceptual_hash"] = str(imagehash.phash(img))
    except (UnidentifiedImageError, OSError) as exc:
        # 扩展名像图片但解码失败 -> corrupt（需人工确认）；否则视为无关文件 -> unsupported
        if path.suffix.lower() in IMAGE_EXTS:
            record["quality_status"] = "corrupt"
            record["notes"] = f"decode_failed: {exc.__class__.__name__}"
        return record

    aspect = width / height
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
    notes = []
    if min(width, height) < MIN_SIDE:
        notes.append(f"too_small(<{MIN_SIDE})")
    if max(aspect, 1 / aspect) > EXTREME_ASPECT:
        notes.append(f"extreme_aspect(>{EXTREME_ASPECT})")
    record["notes"] = ";".join(notes)
    return record


def scan_directory(
    input_dir: Path, previous: pl.DataFrame | None = None
) -> pl.DataFrame:
    """递归扫描目录下所有图片，生成符合 MANIFEST_SCHEMA 的清单。

    path 记录相对输入目录的路径，保证清单与目录移动无关。
    previous 传入上一次的 manifest 时做增量扫描：path 相同且
    file_size + mtime 均未变化的文件直接复用旧记录（含 phash），
    只重新扫描新增或被修改的文件。
    """
    files = sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and not any(
            part in RESERVED_DIRS for part in p.relative_to(input_dir).parts[:-1]
        )
    )
    rel_paths = [p.relative_to(input_dir).as_posix() for p in files]

    # 增量缓存：path -> 上一行的记录
    prev_by_path: dict[str, dict] = {}
    if previous is not None and previous.height > 0 and "mtime" in previous.columns:
        prev_by_path = {r["path"]: r for r in previous.iter_rows(named=True)}

    def reuse(rel: str, path: Path) -> dict | None:
        prev = prev_by_path.get(rel)
        if prev is None:
            return None
        st = path.stat()
        if prev["file_size"] == st.st_size and prev["mtime"] == st.st_mtime:
            return {name: prev[name] for name in MANIFEST_SCHEMA}
        return None

    def scan_one(item: tuple[Path, str]) -> dict:
        path, rel = item
        record = scan_single_file(path)
        record["path"] = rel
        return record

    cached_rows: dict[str, dict] = {}
    todo: list[tuple[Path, str]] = []
    for path, rel in zip(files, rel_paths):
        row = reuse(rel, path)
        if row is not None:
            cached_rows[rel] = row
        else:
            todo.append((path, rel))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        scanned = iter(pool.map(scan_one, todo))
    records = [cached_rows.get(rel) or next(scanned) for rel in rel_paths]

    return pl.DataFrame(records, schema=MANIFEST_SCHEMA)


def group_near_duplicates(
    manifest: pl.DataFrame, max_distance: int = NEAR_DUP_DISTANCE
) -> list[list[str]]:
    """按 phash 汉明距离对图片做近重复分组（并查集求连通分量）。

    返回路径分组（每组 >= 2 张，组内任意两张距离 <= max_distance 经由
    传递闭包连接）。两两比较为 O(n^2)，几千张图的规模在数秒内可完成。
    """
    ok = manifest.filter(
        (pl.col("quality_status") == "ok") & pl.col("perceptual_hash").is_not_null()
    )
    hashes = [int(h, 16) for h in ok["perceptual_hash"]]
    paths = ok["path"].to_list()

    n = len(paths)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        hi = hashes[i]
        for j in range(i + 1, n):
            if (hi ^ hashes[j]).bit_count() <= max_distance:
                # 按秩不做了，路径压缩足够
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[str]] = {}
    for i, path in enumerate(paths):
        groups.setdefault(find(i), []).append(path)
    return [g for g in groups.values() if len(g) > 1]


def summarize(manifest: pl.DataFrame) -> dict:
    """生成扫描报告：总数、按格式/状态聚合、尺寸统计、待人工确认清单、重复分组。"""
    ok = manifest.filter(pl.col("quality_status") == "ok")
    exact_groups = (
        ok.group_by("image_id")
        .agg(pl.col("path").sort())
        .filter(pl.col("path").list.len() > 1)  # pl.len() 在此上下文是全表行数
        .sort("image_id")["path"]
        .to_list()
    )
    near_groups = group_near_duplicates(manifest)
    report = {
        "total_files": manifest.height,
        "by_quality_status": manifest.group_by("quality_status")
        .len()
        .sort("quality_status")
        .to_dicts(),
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
        # 完全重复：sha256 相同；near: phash 汉明距离 <= NEAR_DUP_DISTANCE
        "duplicate_exact_groups": len(exact_groups),
        "duplicate_exact_paths": exact_groups,
        "duplicate_near_groups": near_groups,
    }
    return report
