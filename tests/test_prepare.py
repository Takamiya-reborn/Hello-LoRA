"""prepare 模块测试：构造带 metadata.jsonl 的最小数据集，验证训练结构与 manifest。"""

import json
from pathlib import Path

import pytest
from PIL import Image

from hello_lora.prepare import build_caption, prepare_dataset


def make_image(path: Path, size: tuple[int, int] = (768, 1024)) -> None:
    Image.new("RGB", size, "red").save(path, format="JPEG")


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """两张有效图 + 一个无元数据的孤儿文件 + 一个有元数据的损坏文件。"""
    d = tmp_path / "sample_artist"
    d.mkdir()
    records = []
    for i in range(2):
        md5 = f"{'0' * i}{'1' * (32 - i)}"  # 彼此不同的假 md5
        make_image(d / f"{md5}.jpg")
        records.append(
            {
                "id": 1000 + i,
                "md5": md5,
                "rating": "s" if i == 0 else "e",
                "score": 10 + i,
                "image_width": 768,
                "image_height": 1024,
                "file_size": 12345,
                "tag_count_general": 5,
                "tag_string_general": "1girl long_hair",
                "tag_string_copyright": "orig",
                "tag_string_artist": "sample_(artist)",
            }
        )
    # 有元数据但内容损坏的文件，用于覆盖解码失败分支
    records.append({"id": 1002, "md5": "2" * 32, "rating": "g", "score": 1})
    (d / "metadata.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8"
    )
    make_image(d / "orphan.jpg")  # 无元数据
    (d / f"{'2' * 32}.jpg").write_bytes(b"not an image")  # 解码失败
    return d


def test_prepare_dataset_basic(dataset_dir: Path) -> None:
    manifest, train_dir, skipped = prepare_dataset(dataset_dir)

    assert manifest.height == 2
    # kohya 概念目录名：数据集名净化 + repeats 前缀
    assert train_dir.name == "1_sample_artist"
    # 图片与同名 caption 成对
    txts = sorted(p.name for p in train_dir.glob("*.txt"))
    assert txts == [p.stem + ".txt" for p in sorted(train_dir.glob("*.jpg"))]
    # 孤儿文件与损坏文件都被跳过且给出原因
    reasons = {name: reason for name, reason in skipped}
    assert reasons["orphan.jpg"] == "无元数据"
    assert reasons["2" * 32 + ".jpg"].startswith("解码失败")


def test_prepare_dataset_rating_filter(dataset_dir: Path) -> None:
    manifest, _, skipped = prepare_dataset(dataset_dir, ratings=["s"])

    assert manifest.height == 1
    assert manifest["rating"][0] == "s"
    assert any("评分过滤" in reason for _, reason in skipped)


def test_build_caption_excludes_artist_by_default() -> None:
    record = {
        "tag_string_general": "1girl long_hair",
        "tag_string_copyright": "orig",
        "tag_string_artist": "sample_(artist)",
    }
    caption = build_caption(record, trigger_word=None, keep_artist=False)

    assert caption == "1girl, long hair, orig"  # 标签逗号分隔，下划线转为空格
    assert "sample" not in caption
    # 触发词加在开头；artist 标签可选保留
    assert build_caption(record, "my_style", True).startswith("my_style, 1girl")
    assert "sample (artist)" in build_caption(record, None, True)
