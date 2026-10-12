"""scan 模块测试：用 Pillow 在临时目录构造各类图片，验证清单字段和状态标记。"""

import io
from pathlib import Path

import polars as pl
import pytest
from PIL import Image

from hello_lora.scan import (
    EXTREME_ASPECT,
    MIN_SIDE,
    scan_directory,
    summarize,
)


def make_image(
    path: Path, size: tuple[int, int], fmt: str = "PNG", color: str = "red"
) -> None:
    img = Image.new("RGB", size, color)
    img.save(path, format=fmt)


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    """构造包含正常、过小、极端比例、损坏图片的目录（非图片文件不进清单）。"""
    make_image(tmp_path / "ok.png", (1024, 1024))
    make_image(tmp_path / "small.png", (MIN_SIDE - 1, MIN_SIDE))
    make_image(
        tmp_path / "wide.png",
        (int(MIN_SIDE * EXTREME_ASPECT * 2), MIN_SIDE),
        color="blue",
    )
    make_image(tmp_path / "photo.webp", (768, 1024), fmt="WEBP", color="green")

    corrupt = Image.new("RGB", (64, 64))
    buf = io.BytesIO()
    corrupt.save(buf, format="PNG")
    (tmp_path / "corrupt.png").write_bytes(buf.getvalue()[:20])

    (tmp_path / "not_image.txt").write_text("hello")
    return tmp_path


def test_manifest_schema_and_rows(sample_dir: Path) -> None:
    manifest = scan_directory(sample_dir)
    assert manifest.height == 5
    assert manifest["image_id"].null_count() == 0
    assert manifest["perceptual_hash"].null_count() == 1  # 只有 corrupt 没有 phash
    # 非图片文件不进 manifest
    assert set(manifest["path"]) == {
        "corrupt.png",
        "ok.png",
        "photo.webp",
        "small.png",
        "wide.png",
    }


def test_quality_status(sample_dir: Path) -> None:
    manifest = scan_directory(sample_dir)
    by_path = dict(zip(manifest["path"], manifest["quality_status"]))
    assert by_path["ok.png"] == "ok"
    assert by_path["photo.webp"] == "ok"
    assert by_path["corrupt.png"] == "corrupt"


def test_review_flags(sample_dir: Path) -> None:
    manifest = scan_directory(sample_dir)
    notes = dict(zip(manifest["path"], manifest["notes"]))
    assert "too_small" in notes["small.png"]
    assert "extreme_aspect" in notes["wide.png"]
    assert notes["ok.png"] == ""
    assert notes["corrupt.png"].startswith("decode_failed")


def test_exact_duplicate_shares_image_id(sample_dir: Path) -> None:
    make_image(sample_dir / "copy.png", (1024, 1024))  # 与 ok.png 内容相同的重复图
    manifest = scan_directory(sample_dir)
    ids = dict(zip(manifest["path"], manifest["image_id"]))
    assert ids["copy.png"] == ids["ok.png"]
    report = summarize(manifest)
    assert report["duplicate_exact_groups"] == 1
    assert report["duplicate_exact_paths"] == [["copy.png", "ok.png"]]


def test_near_duplicate_group(sample_dir: Path) -> None:
    # 同一张图两种编码：sha256 不同（非完全重复），phash 相同（近重复）
    img = Image.new("RGB", (1024, 1024), "purple")
    img.save(sample_dir / "near_a.png", format="PNG")
    img.save(sample_dir / "near_b.jpg", format="JPEG", quality=95)
    report = summarize(scan_directory(sample_dir))
    containing = [g for g in report["duplicate_near_groups"] if "near_a.png" in g]
    assert containing, f"未检出近重复组: {report['duplicate_near_groups']}"
    assert set(containing[0]) >= {"near_a.png", "near_b.jpg"}


def test_reserved_dirs_and_metadata_skipped(sample_dir: Path) -> None:
    # 流水线保留目录 / 元数据文件不应被扫进清单（否则 prepare 后重扫会误报重复）
    (sample_dir / "metadata.jsonl").write_text("{}")
    (sample_dir / "train" / "1_artist").mkdir(parents=True)
    make_image(sample_dir / "train" / "1_artist" / "nested.png", (256, 256))
    (sample_dir / "manifest").mkdir()
    make_image(sample_dir / "manifest" / "x.png", (256, 256))

    manifest = scan_directory(sample_dir)
    assert manifest.height == 5
    assert not any(p.startswith(("train/", "manifest/")) for p in manifest["path"])


def test_incremental_rescan(sample_dir: Path) -> None:
    first = scan_directory(sample_dir)
    # 无变化时全量复用，结果一致
    second = scan_directory(sample_dir, previous=first)
    assert second.equals(first)

    # 文件被替换后该行被重新扫描（尺寸变化保证 size+mtime 失效）
    (sample_dir / "ok.png").unlink()
    make_image(sample_dir / "ok.png", (768, 768), color="cyan")
    third = scan_directory(sample_dir, previous=second)

    assert "photo.webp" in second["path"]  # sanity check
    (sample_dir / "photo.webp").unlink()
    fourth = scan_directory(sample_dir, previous=third)
    assert third.height == 5
    assert fourth.height == 4
    assert fourth.filter(pl.col("path") == "ok.png")["width"][0] == 768


def test_summarize_report(sample_dir: Path) -> None:
    manifest = scan_directory(sample_dir)
    report = summarize(manifest)
    assert report["total_files"] == 5
    assert report["size_stats"]["width_max"] >= 1024
    assert set(report["needs_review"]) == {"small.png", "wide.png", "corrupt.png"}
