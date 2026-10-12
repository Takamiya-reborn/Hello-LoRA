"""tag 模块测试：标签选择、CSV 解析、预处理与文件枚举（不下载模型，不跑推理）。"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hello_lora.tag import (
    CATEGORY_CHARACTER,
    CATEGORY_GENERAL,
    CATEGORY_RATING,
    build_caption,
    format_caption,
    iter_image_files,
    load_labels,
    preprocess_image,
    select_tags,
)

# 模拟 selected_tags.csv 的行序：4 个 rating -> general -> artist -> copyright -> character
LABELS = [
    "general",
    "sensitive",
    "questionable",
    "explicit",
    "1girl",
    "long_hair",
    "some_artist",
    "some_series",
    "rei_ayanami",
]
CATEGORIES = [9, 9, 9, 9, 0, 0, 1, 3, 4]


def test_load_labels(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    csv_path.write_text(
        "tag_id,name,category,count\n0,1girl,0,1000\n1,rei_ayanami,4,50\n",
        encoding="utf-8",
    )
    labels, categories = load_labels(csv_path)
    assert labels == ["1girl", "rei_ayanami"]
    assert categories == [CATEGORY_GENERAL, CATEGORY_CHARACTER]


def test_load_labels_bad_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    csv_path.write_text("wrong,header,here\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_labels(csv_path)


def test_select_tags_thresholds_and_order() -> None:
    # general 0.36 过线、artist 0.8 再高也不进 caption、character 0.95 过 character 阈值
    prob = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.36, 0.8, 0.2, 0.95])
    general, character, rating, rating_prob = select_tags(LABELS, CATEGORIES, prob)

    assert general == ["1girl", "long_hair"]  # 按置信度降序
    assert character == ["rei_ayanami"]
    assert rating == "explicit"
    assert rating_prob == pytest.approx(0.4)


def test_select_tags_custom_threshold() -> None:
    prob = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.36, 0.8, 0.2, 0.95])
    general, character, _, _ = select_tags(
        LABELS, CATEGORIES, prob, general_threshold=0.5
    )
    assert general == ["1girl"]  # long_hair 0.36 被滤掉
    assert character == ["rei_ayanami"]


def test_select_tags_rating_argmax() -> None:
    prob = np.zeros(len(LABELS))
    prob[:4] = [0.7, 0.1, 0.1, 0.1]
    general, character, rating, rating_prob = select_tags(LABELS, CATEGORIES, prob)
    assert general == [] and character == []  # 无 general/character 过线
    assert rating == "general"
    assert rating_prob == pytest.approx(0.7)


def test_build_caption_purpose() -> None:
    # style：只有 general 标签；character：角色标签放最前
    assert build_caption(["1girl", "long hair"], ["rei ayanami"], "style") == (
        "1girl, long hair"
    )
    assert build_caption(["1girl", "long hair"], ["rei ayanami"], "character") == (
        "rei ayanami, 1girl, long hair"
    )


def test_format_caption() -> None:
    assert format_caption(["long_hair", "1girl"]) == "long hair, 1girl"
    assert format_caption([]) == ""


def test_preprocess_image_rgb() -> None:
    img = Image.new("RGB", (100, 60), (0, 0, 255))
    arr = preprocess_image(img, 64)
    assert arr.shape == (64, 64, 3)
    assert arr.dtype == np.float32
    # 中心是原色，BGR 顺序 -> 红色通道在前
    assert arr[32, 32].tolist() == [255.0, 0.0, 0.0]
    # 白边补方：上下各补 20px 再缩放，边缘应为白色
    assert arr[0, 32].tolist() == [255.0, 255.0, 255.0]


def test_preprocess_image_transparent_on_white() -> None:
    img = Image.new("RGBA", (100, 60), (255, 0, 0, 0))  # 全透明的红色
    arr = preprocess_image(img, 64)
    # 透明区域合成到白底
    assert arr[0, 32].tolist() == [255.0, 255.0, 255.0]
    assert arr[32, 32].tolist() == [255.0, 255.0, 255.0]


def test_iter_image_files_skips_reserved(tmp_path: Path) -> None:
    Image.new("RGB", (64, 64)).save(tmp_path / "a.jpg")
    (tmp_path / "sub").mkdir()
    Image.new("RGB", (64, 64)).save(tmp_path / "sub" / "b.png")
    (tmp_path / "manifest").mkdir()
    Image.new("RGB", (64, 64)).save(tmp_path / "manifest" / "x.png")
    (tmp_path / "train" / "1_x").mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(tmp_path / "train" / "1_x" / "y.webp")
    (tmp_path / "notes.txt").write_text("hello")

    files = iter_image_files(tmp_path)
    assert files == ["a.jpg", "sub/b.png"]
