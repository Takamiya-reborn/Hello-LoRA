"""prepare 模块测试：构造带 metadata.jsonl 的最小数据集，验证训练结构与 manifest。"""

import json
from pathlib import Path

import polars as pl
import pytest
from PIL import Image

from hello_lora.prepare import assign_split, build_caption, prepare_dataset


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


def test_prepare_prefers_tag_txt(dataset_dir: Path) -> None:
    # 打标产物优先：同名 .txt 存在时 caption 直接采用其内容（触发词仍前缀）
    md5 = "1" * 32
    (dataset_dir / f"{md5}.txt").write_text("wd tag, another tag", encoding="utf-8")
    _, train_dir, _ = prepare_dataset(dataset_dir, trigger_word="my_style")

    caption = (train_dir / f"{md5}.txt").read_text(encoding="utf-8")
    assert caption == "my_style, wd tag, another tag"
    # 未打标的图片仍走 metadata.jsonl 构建
    other_md5 = "0" + "1" * 31  # fixture 中第二张图的假 md5
    other = (train_dir / f"{other_md5}.txt").read_text(encoding="utf-8")
    assert other == "my_style, 1girl, long hair, orig"


def test_prepare_without_metadata_but_with_txt(tmp_path: Path) -> None:
    # Kaggle 等无元数据的数据集：只有打标 txt 也能 prepare，booru 元数据列为 null
    d = tmp_path / "kaggle_ds"
    d.mkdir()
    make_image(d / "a.jpg")
    (d / "a.txt").write_text("1girl, solo", encoding="utf-8")

    manifest, train_dir, skipped = prepare_dataset(d)
    assert manifest.height == 1
    assert skipped == []
    assert manifest["md5"][0] is None
    assert manifest["rating"][0] is None
    assert manifest["width"][0] == 768  # 尺寸来自实际解码而非元数据
    assert (train_dir / "a.txt").read_text(encoding="utf-8") == "1girl, solo"


def test_prepare_without_caption_source(tmp_path: Path) -> None:
    d = tmp_path / "empty_ds"
    d.mkdir()
    make_image(d / "a.jpg")
    with pytest.raises(SystemExit):
        prepare_dataset(d)


def test_assign_split_deterministic() -> None:
    # 同一 image_id 归属稳定，与调用次数和文件顺序无关
    image_id = "a" * 16
    assert assign_split(image_id, 0.0) == "train"
    assert assign_split(image_id, 0.5) == assign_split(image_id, 0.5)
    expected = "val" if int(image_id, 16) % 10_000 < 5_000 else "train"
    assert assign_split(image_id, 0.5) == expected


def test_prepare_val_ratio_out_of_range(dataset_dir: Path) -> None:
    with pytest.raises(SystemExit):
        prepare_dataset(dataset_dir, val_ratio=1.0)
    with pytest.raises(SystemExit):
        prepare_dataset(dataset_dir, val_ratio=-0.1)


def test_prepare_val_split(tmp_path: Path) -> None:
    # 20 张互不相同的图，0.5 比例切分应同时产生 train 和 val
    d = tmp_path / "split_ds"
    d.mkdir()
    for i in range(20):
        Image.new("RGB", (64, 64), (i * 12 % 255, 30, 30)).save(d / f"img{i:02d}.jpg")
        (d / f"img{i:02d}.txt").write_text("1girl, solo", encoding="utf-8")

    manifest, train_dir, _ = prepare_dataset(d, val_ratio=0.5)

    # split 列与确定性规则一致，且两种 split 都出现了
    for row in manifest.iter_rows(named=True):
        assert row["split"] == assign_split(row["image_id"], 0.5)
    assert set(manifest["split"]) == {"train", "val"}

    # val 图在 <dataset>/val/ 且不在 train 目录，caption 同样成对
    val_dir = d / "val"
    val_rows = manifest.filter(pl.col("split") == "val")
    assert val_rows.height > 0
    for row in val_rows.iter_rows(named=True):
        path = Path(row["path"])
        assert path.parent == val_dir
        assert not (train_dir / path.name).exists()
        assert (val_dir / f"{path.stem}.txt").is_file()

    # 重跑归属不变（确定性切分）
    manifest2, _, _ = prepare_dataset(d, val_ratio=0.5)
    assert manifest2.sort("image_id")["split"].to_list() == manifest.sort("image_id")["split"].to_list()


def test_prepare_backfills_scan_split(dataset_dir: Path) -> None:
    # 预写一个 split 列全空的假 scan.parquet，prepare 后按 image_id 回填
    manifest, _, _ = prepare_dataset(dataset_dir)
    scan = pl.DataFrame(
        {
            "image_id": manifest["image_id"].to_list(),
            "path": [f"{i}.jpg" for i in range(manifest.height)],
            "split": [None] * manifest.height,
        },
        schema={"image_id": pl.String, "path": pl.String, "split": pl.String},
    )
    scan.write_parquet(dataset_dir / "manifest" / "scan.parquet")

    prepare_dataset(dataset_dir)  # 再跑一次，触发回填

    scan_after = pl.read_parquet(dataset_dir / "manifest" / "scan.parquet")
    by_id = dict(zip(manifest["image_id"], manifest["split"]))
    for row in scan_after.iter_rows(named=True):
        assert row["split"] == by_id[row["image_id"]]
