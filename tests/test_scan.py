"""scan 模块测试：用 Pillow 在临时目录构造各类图片，验证清单字段和状态标记。"""

import io
from pathlib import Path

import pytest
from PIL import Image

from hello_lora.scan import EXTREME_ASPECT, MIN_SIDE, scan_directory, summarize


def make_image(
    path: Path, size: tuple[int, int], fmt: str = "PNG", color: str = "red"
) -> None:
    img = Image.new("RGB", size, color)
    img.save(path, format=fmt)


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    """构造包含正常、过小、极端比例、损坏、无关文件的目录。"""
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
    assert manifest.height == 6
    assert manifest["image_id"].null_count() == 0
    assert set(manifest["path"]) == {
        "corrupt.png",
        "not_image.txt",
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
    assert by_path["not_image.txt"] == "unsupported"


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


def test_summarize_report(sample_dir: Path) -> None:
    manifest = scan_directory(sample_dir)
    report = summarize(manifest)
    assert report["total_files"] == 6
    assert report["size_stats"]["width_max"] >= 1024
    assert set(report["needs_review"]) == {"small.png", "wide.png", "corrupt.png"}
