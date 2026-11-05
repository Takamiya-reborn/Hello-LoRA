"""阶段四：WD14 打标，用 ONNX 模型为图片生成 kohya 同名 .txt 标签文件。

产物归属数据集目录自身，保持自包含:

    <dataset_dir>/
    ├─ *.jpg / *.png / *.webp
    ├─ <stem>.txt                  # 每张图一个标签文件（prepare 阶段优先使用）
    ├─ manifest/tag.parquet        # 打标清单（rating、标签数等）
    └─ manifest/scan.parquet       # 存在时回填 caption_status 列

模型来自 Hugging Face 的 SmilingWolf 系列（ONNX），缓存到 assets/models/onnx/<repo>，
本地齐全时不再联网；国内网络可设 HF_ENDPOINT=https://hf-mirror.com 走镜像。
预处理与 sd-scripts 的 tag_images_by_wd14_tagger.py 保持同一口径：
透明通道合成到白底 -> 白边补成正方形 -> 缩放到模型输入尺寸 -> RGB 转 BGR。
"""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort
import polars as pl
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm

from hello_lora.scan import IMAGE_EXTS, RESERVED_DIRS

# SmilingWolf 的 ONNX 模型仓库：内置别名 -> HF repo id
MODEL_REPOS = {
    "wd-vit-v3": "SmilingWolf/wd-vit-tagger-v3",
    "wd-swinv2-v3": "SmilingWolf/wd-swinv2-tagger-v3",
    "wd-eva02-large-v3": "SmilingWolf/wd-eva02-large-tagger-v3",
    "wd-convnextv2-v2": "SmilingWolf/wd-convnextv2-tagger-v2",
    "wd-swinv2-v2": "SmilingWolf/wd-swinv2-tagger-v2",
    "wd-moat-v2": "SmilingWolf/wd-v1-4-moat-tagger-v2",
}
DEFAULT_MODEL = (
    "wd-vit-v3"  # v3 系列：精度与推理速度均衡（主环境 onnxruntime 为 CPU 版）
)

# selected_tags.csv 的 category 编码：9=rating，0=general，4=character
# （与 sd-scripts 的 wd14 打标脚本一致；1=artist / 3=copyright 不进 caption）
CATEGORY_RATING = 9
CATEGORY_GENERAL = 0
CATEGORY_CHARACTER = 4

GENERAL_THRESHOLD = 0.35  # wd14 tagger 社区默认
CHARACTER_THRESHOLD = 0.85

# 训练目标决定 caption 组成：
# style  画风/笔触：只保留 general 标签（笔触、线条、构图都在这里），剔除 character 标签，
#        避免把角色身份绑进画风；触发词在 prepare 阶段用 --trigger-word 绑定。
# character 角色：character 标签（角色名/服装变体）放 caption 最前，紧跟特征 general 标签；
#        角色特征细节多，建议把 --threshold 降到 0.25 左右多保留细节。
PURPOSES = ("style", "character")
DEFAULT_PURPOSE = "style"

BATCH_SIZE = 8
MAX_WORKERS = 8  # 并行解码线程数
FALLBACK_SIZE = 448  # 模型输入尺寸读不出来时的兜底值

TAG_SCHEMA = {
    "path": pl.String,  # 相对数据集目录的路径，与 scan 清单口径一致
    "purpose": pl.String,  # style / character，本次打标的训练目标
    "rating": pl.String,  # rating 行 argmax 的标签（general/sensitive/…）
    "rating_prob": pl.Float32,
    "tags": pl.String,  # 写入 txt 的最终 caption（booru 下划线已转空格）
    "n_tags": pl.Int32,
    "status": pl.String,  # ok / corrupt
    "notes": pl.String,
}


def default_model_dir() -> Path:
    """模型缓存目录：仓库内 assets/models/onnx，脱离仓库运行时退回相对路径。"""
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file():
        return root / "assets" / "models" / "onnx"
    return Path("assets/models/onnx")


def ensure_model(model_root: Path, repo: str) -> Path:
    """返回模型目录；缺 model.onnx 或 selected_tags.csv 时从 HF 下载补齐。

    huggingface_hub 在此函数内才 import（而非模块顶层）：CLI 启动时先把
    profile 的 [network] hf_endpoint 写进环境变量，这里的首个 HF 调用才能
    走到镜像。
    """
    from huggingface_hub import hf_hub_download

    model_dir = model_root / repo.replace("/", "_")
    onnx_path = model_dir / "model.onnx"
    csv_path = model_dir / "selected_tags.csv"
    if onnx_path.is_file() and csv_path.is_file():
        return model_dir
    for filename in ("model.onnx", "selected_tags.csv"):
        hf_hub_download(repo_id=repo, filename=filename, local_dir=model_dir)
    return model_dir


def create_session(onnx_path: Path) -> tuple[ort.InferenceSession, int]:
    """创建 ONNX 会话，返回 (session, 模型输入边长)。"""
    available = ort.get_available_providers()
    providers = [
        p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available
    ]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    shape = session.get_inputs()[0].shape  # 形如 ['N', 448, 448, 3]
    size = shape[1] if isinstance(shape[1], int) and shape[1] > 0 else FALLBACK_SIZE
    return session, size


def load_labels(csv_path: Path) -> tuple[list[str], list[int]]:
    """解析 selected_tags.csv，返回 (标签名, category)，顺序即输出向量顺序。"""
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header[:3] != ["tag_id", "name", "category"]:
            raise SystemExit(f"selected_tags.csv 格式不符: {header}")
        names, categories = [], []
        for row in reader:
            names.append(row[1])
            categories.append(int(row[2]))
    return names, categories


def select_tags(
    labels: list[str],
    categories: list[int],
    prob: np.ndarray,
    general_threshold: float = GENERAL_THRESHOLD,
    character_threshold: float = CHARACTER_THRESHOLD,
) -> tuple[list[str], list[str], str, float]:
    """从单张图的预测向量里选出 caption 标签与 rating。

    返回 (general 标签, character 标签, rating, rating_prob)。rating 取 category=9
    行的 argmax；general / character 分别按各自阈值过滤后按置信度降序输出
    （靠前的标签对训练影响更大）。artist / copyright 等不进 caption。
    """
    prob = np.asarray(prob)
    rating_idx = [i for i, c in enumerate(categories) if c == CATEGORY_RATING]
    rating_pos = int(np.argmax(prob[rating_idx]))
    rating = labels[rating_idx[rating_pos]]
    rating_prob = float(prob[rating_idx[rating_pos]])

    general: list[tuple[float, str]] = []
    character: list[tuple[float, str]] = []
    for i, category in enumerate(categories):
        if category == CATEGORY_GENERAL and prob[i] >= general_threshold:
            general.append((float(prob[i]), labels[i]))
        elif category == CATEGORY_CHARACTER and prob[i] >= character_threshold:
            character.append((float(prob[i]), labels[i]))
    general.sort(key=lambda item: item[0], reverse=True)
    character.sort(key=lambda item: item[0], reverse=True)
    return (
        [name for _, name in general],
        [name for _, name in character],
        rating,
        rating_prob,
    )


def build_caption(
    general_tags: list[str], character_tags: list[str], purpose: str
) -> str:
    """按训练目标组 caption：character 模式把角色标签放在最前。"""
    tags = character_tags + general_tags if purpose == "character" else general_tags
    return format_caption(tags)


def format_caption(tags: list[str]) -> str:
    """booru 下划线转空格，逗号分隔（与 prepare.build_caption 口径一致）。"""
    return ", ".join(tag.replace("_", " ") for tag in tags)


def preprocess_image(img: Image.Image, size: int) -> np.ndarray:
    """按训练口径预处理：白底合成透明 -> 白边补方形 -> 缩放 -> BGR float32 (NHWC 单张)。"""
    if img.mode in ("RGBA", "LA", "P") or "transparency" in img.info:
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    else:
        img = img.convert("RGB")
    img = ImageOps.pad(
        img,
        (size, size),
        method=Image.Resampling.BICUBIC,
        color=(255, 255, 255),
        centering=(0.5, 0.5),
    )
    arr = np.asarray(img, dtype=np.float32)
    return np.ascontiguousarray(arr[:, :, ::-1])  # RGB -> BGR


def iter_image_files(input_dir: Path) -> list[str]:
    """递归列出图片文件的相对路径，跳过流水线保留目录（与 scan 口径一致）。"""
    return sorted(
        p.relative_to(input_dir).as_posix()
        for p in input_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and not any(
            part in RESERVED_DIRS for part in p.relative_to(input_dir).parts[:-1]
        )
    )


def tag_dataset(
    dataset_dir: Path,
    repo: str = MODEL_REPOS[DEFAULT_MODEL],
    purpose: str = DEFAULT_PURPOSE,
    general_threshold: float = GENERAL_THRESHOLD,
    character_threshold: float = CHARACTER_THRESHOLD,
    batch_size: int = BATCH_SIZE,
    overwrite: bool = False,
    model_root: Path | None = None,
) -> tuple[pl.DataFrame, list[tuple[str, str]]]:
    """给数据集打标，返回 (tag manifest, 跳过列表)。

    标签写入图片同目录的 <stem>.txt（kohya 约定）；已存在的 txt 默认跳过，增量可重跑。
    purpose 决定 caption 组成（style 只留 general，character 把角色标签放最前），
    换目标重标需加 --overwrite。
    """
    root = model_root or default_model_dir()
    model_dir = ensure_model(root, repo)
    session, size = create_session(model_dir / "model.onnx")
    print(
        f"模型: {repo} ({size}x{size}), 推理后端: {'+'.join(session.get_providers())}"
    )
    labels, categories = load_labels(model_dir / "selected_tags.csv")
    input_name = session.get_inputs()[0].name
    if purpose not in PURPOSES:
        raise SystemExit(f"未知训练目标: {purpose}（可选 {', '.join(PURPOSES)}）")

    todo = [
        rel
        for rel in iter_image_files(dataset_dir)
        if overwrite or not (dataset_dir / Path(rel).with_suffix(".txt")).is_file()
    ]
    skipped: list[tuple[str, str]] = []

    def load_one(rel: str) -> tuple[str, np.ndarray | None, str]:
        try:
            with Image.open(dataset_dir / rel) as img:
                return rel, preprocess_image(img, size), ""
        except (UnidentifiedImageError, OSError) as exc:
            return rel, None, f"decode_failed: {exc.__class__.__name__}"

    rows: list[dict] = []
    batches = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]
    with (
        ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool,
        tqdm(batches, desc="WD14 打标", unit="批", smoothing=0.1) as bar,
    ):
        for batch in bar:
            loaded = list(pool.map(load_one, batch))
            inputs, valid = [], []
            for rel, arr, error in loaded:
                if arr is None:
                    skipped.append((rel, error))
                else:
                    inputs.append(arr)
                    valid.append(rel)
            if not inputs:
                continue
            probs = session.run(None, {input_name: np.stack(inputs)})[0]
            for rel, prob in zip(valid, probs):
                general_tags, character_tags, rating, rating_prob = select_tags(
                    labels, categories, prob, general_threshold, character_threshold
                )
                caption = build_caption(general_tags, character_tags, purpose)
                txt = dataset_dir / Path(rel).with_suffix(".txt")
                txt.write_text(caption + "\n", encoding="utf-8")
                rows.append(
                    {
                        "path": rel,
                        "purpose": purpose,
                        "rating": rating,
                        "rating_prob": round(rating_prob, 4),
                        "tags": caption,
                        "n_tags": len(general_tags) + len(character_tags),
                        "status": "ok",
                        "notes": "",
                    }
                )
            bar.set_postfix(n_tags=rows[-1]["n_tags"] if rows else 0)

    # 清单按 path 做增量合并：只保留同一训练目标的旧行（本次没碰的已打标文件）
    manifest_dir = dataset_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = pl.DataFrame(rows, schema=TAG_SCHEMA)
    tag_path = manifest_dir / "tag.parquet"
    if tag_path.is_file():
        previous = pl.read_parquet(tag_path)
        if "purpose" in previous.columns:
            previous = previous.filter(pl.col("purpose") == purpose)
        fresh_paths = set(manifest["path"])
        manifest = pl.concat(
            [previous.filter(~pl.col("path").is_in(fresh_paths)), manifest]
        )
    manifest.write_parquet(tag_path)
    _backfill_scan_status(manifest_dir, manifest)

    return manifest, skipped


def _backfill_scan_status(manifest_dir: Path, tag_manifest: pl.DataFrame) -> None:
    """scan 清单存在时，把 caption_status 列回填为本次打标状态。"""
    scan_path = manifest_dir / "scan.parquet"
    if not scan_path.is_file() or tag_manifest.height == 0:
        return
    scan_manifest = pl.read_parquet(scan_path)
    if "caption_status" not in scan_manifest.columns:
        return
    updates = tag_manifest.select(
        pl.col("path"), pl.col("status").alias("caption_status_from_tag")
    )
    scan_manifest = (
        scan_manifest.join(updates, on="path", how="left")
        .with_columns(
            pl.coalesce("caption_status_from_tag", "caption_status").alias(
                "caption_status"
            )
        )
        .drop("caption_status_from_tag")
        .sort("path")
    )
    scan_manifest.write_parquet(scan_path)
