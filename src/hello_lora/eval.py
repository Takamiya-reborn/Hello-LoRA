"""阶段八：拟合评估，用 CLIP 指标判断各 checkpoint 过拟合 / 欠拟合。

对实验目录里的每个 epoch checkpoint 用固定 prompt 事后生成样例图
（diffusers，inference 依赖组），再算 CLIP 指标，产出:

    assets/experiments/<id>/
    ├─ prompts.txt                  # 可选：固定评估提示词（无则用内置默认）
    ├─ manifest/eval.parquet        # 每 checkpoint × 每 prompt × 每 seed 一行
    ├─ manifest/eval.report.json    # 逐 checkpoint 汇总指标 + 判定 + 推荐 epoch
    ├─ manifest/clip_ref_cache.npz  # 参考图 embedding 缓存（按 image_id）
    └─ samples/eval/<checkpoint>/<prompt_id>_seed<seed>.png

指标口径（base = 底模无 LoRA 的基线，e = 某 epoch）：

- style_sim_train   样例与训练参考图 embedding 余弦的均值（画风保真）；
                    低于 base + STYLE_GAIN_MIN -> 欠拟合（LoRA 没起效）
- style_sim_val     与验证集参考图均值的余弦；gap = train - val 随 epoch
                    增大 -> 过拟合信号（无验证集时为 null，判定降级）
- clip_t            图文余弦（prompt 遵循度）；后期下跌 -> 语义可控性被吃掉
- diversity         checkpoint 内样例两两余弦的均值（越低越同质）；
                    相对 base 跌幅 > DIVERSITY_DROP -> 输出塌缩 / 过拟合
- nearest_train_sim 样例与最近训练图的余弦；> MEMORIZATION_MAX -> 复刻训练图

阈值均为经验规则（启发式），报告里逐 epoch 列表供人工复核；
"style 上升趋平、clip_t 开始回落"的拐点通常就是甜点。

torch / transformers / hello_lora.generate 只在本模块函数内惰性 import，
保证无 torch 的主环境 `import hello_lora.cli` 不受影响。
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

# CLIP 打分模型：内置别名 -> HF repo id；ViT-L 是 CLIP-I / CLIP-T 文献常用
# 口径且权重仅 ~1.7GB，6GB 显存友好；vit-bigg 是 SDXL 二级文本编码器同款
CLIP_REPOS = {
    "vit-l": "openai/clip-vit-large-patch14",
    "vit-bigg": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
}
DEFAULT_CLIP = "vit-l"

# 判定阈值（经验规则，报告注明为启发式）
STYLE_GAIN_MIN = 0.02  # style 相对 base 增益低于此 -> 欠拟合 / LoRA 无效
MEMORIZATION_MAX = 0.9  # 最近训练图余弦相似度超过此 -> 记忆化风险
DIVERSITY_DROP = 0.05  # 相对 base 多样性跌幅超过此 -> 输出塌缩
OVERFIT_GAP = 0.03  # 泛化 gap 相对 base 增大超过此 -> 过拟合（需验证集）
PROMPT_CLIP_FLOOR = 0.0  # 候选 epoch 的 clip_t 相对 base 允许的跌幅

# 有验证集时至少这么多张 val 图，泛化 gap 才被认为可信
MIN_VAL_IMAGES = 5

# 生成默认值；分辨率取 768 而非训练的 1024：速度快约一半，CLIP 指标对
# 分辨率不敏感。fp8 只用于训练侧，推理走 fp16 + cpu offload（docstring 注明）
DEFAULT_BASE_SEED = 42
DEFAULT_RESOLUTION = 768
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 7.0
CLIP_BATCH_SIZE = 16  # CLIP 编码批大小（图像 / 文本同用）

# 内置评估提示词：kind=style 是训练集常见主体域（衡量画风习得），
# kind=unseen 是训练集没有的主体/场景（衡量泛化，过拟合时率先崩）。
# 不含触发词——触发词训练的实验请在 assets/experiments/<id>/prompts.txt 自定义。
DEFAULT_PROMPTS = [
    {"prompt_id": "p01", "kind": "style", "prompt": "1girl, long hair, upper body, looking at viewer, classroom, smile"},
    {"prompt_id": "p02", "kind": "style", "prompt": "1girl, animal ears, sitting, outdoors, blue sky, cloud"},
    {"prompt_id": "p03", "kind": "style", "prompt": "1girl, swimsuit, beach, ocean, summer, standing"},
    {"prompt_id": "p04", "kind": "style", "prompt": "1girl, dress, jewelry, night, candlelight, indoors"},
    {"prompt_id": "p05", "kind": "style", "prompt": "1girl, long hair, wind, field, sunset, full body"},
    {"prompt_id": "p06", "kind": "unseen", "prompt": "1girl, astronaut suit, outer space, stars"},
    {"prompt_id": "p07", "kind": "unseen", "prompt": "1girl, victorian dress, ballroom, chandelier"},
    {"prompt_id": "p08", "kind": "unseen", "prompt": "cat, sitting on windowsill, rain, window"},
]

# 一行一条：<prompt_id>|<kind>|<prompt>；# 开头为注释
PROMPT_KINDS = ("style", "unseen")

EVAL_SCHEMA = {
    "checkpoint": pl.String,  # "base" 或 output/ 下文件名 stem
    "epoch": pl.Int32,  # 0 = base 基线
    "prompt_id": pl.String,
    "prompt": pl.String,
    "kind": pl.String,  # style / unseen
    "seed": pl.Int64,
    "image": pl.String,  # 样例图相对实验目录的路径
    "clip_t": pl.Float32,  # 图文余弦（prompt 遵循度）
    "style_sim_train": pl.Float32,  # 与训练参考图均值的余弦（画风保真）
    "style_sim_val": pl.Float32,  # 与验证集参考图均值的余弦；无 val 时 null
    "nearest_train_sim": pl.Float32,  # 与最近训练图的余弦（记忆化）
    "notes": pl.String,
}

SAMPLES_DIR_NAME = "samples/eval"
EPOCH_PATTERN = re.compile(r"-(\d{6})\.safetensors$")


def default_model_dir() -> Path:
    """CLIP 模型缓存目录：仓库内 assets/models/clip，脱离仓库退回相对路径。"""
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file():
        return root / "assets" / "models" / "clip"
    return Path("assets/models/clip")


def ensure_clip_model(model_root: Path, repo: str) -> Path:
    """返回本地模型目录；缺 config 或权重时从 HF 下载补齐。

    tokenizer 文件通常已预置在 assets/models/clip/ 下，hf_hub_download 对
    已存在且校验一致的文件会直接跳过；国内网络可设 HF_ENDPOINT 走镜像。
    """
    from huggingface_hub import hf_hub_download

    model_dir = model_root / repo.replace("/", "_")
    weights = ("model.safetensors", "pytorch_model.bin", "open_clip_pytorch_model.bin")
    if (
        (model_dir / "config.json").is_file()
        and (model_dir / "preprocessor_config.json").is_file()
        and any((model_dir / w).is_file() for w in weights)
    ):
        return model_dir
    # config / tokenizer / 预处理器文件逐个补齐，个别仓库缺某文件时忽略 404
    for filename in (
        "config.json",
        "preprocessor_config.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.json",
        "special_tokens_map.json",
    ):
        try:
            hf_hub_download(repo_id=repo, filename=filename, local_dir=model_dir)
        except Exception:  # noqa: BLE001 - 单文件缺失不应中断
            pass
    for filename in weights:  # 按优先级取第一个能下载到的权重
        try:
            hf_hub_download(repo_id=repo, filename=filename, local_dir=model_dir)
            break
        except Exception:  # noqa: BLE001
            continue
    if not (model_dir / "config.json").is_file() or not any(
        (model_dir / w).is_file() for w in weights
    ):
        raise SystemExit(f"CLIP 模型下载失败: {repo} -> {model_dir}")
    return model_dir


def parse_prompt_line(line: str) -> dict | None:
    """解析一行提示词 `<id>|<kind>|<text>`；空行 / # 注释返回 None。"""
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    parts = text.split("|", 2)
    if len(parts) != 3 or not parts[0].strip() or not parts[2].strip():
        raise SystemExit(f"提示词格式不符（应为 <id>|<kind>|<text>）: {line.strip()}")
    kind = parts[1].strip()
    if kind not in PROMPT_KINDS:
        raise SystemExit(f"未知提示词类别 {kind}（可选 {', '.join(PROMPT_KINDS)}）")
    return {
        "prompt_id": parts[0].strip(),
        "kind": kind,
        "prompt": parts[2].strip(),
    }


def load_prompts(path: Path | None) -> list[dict]:
    """加载评估提示词；文件不存在或未指定时返回内置默认。"""
    if path is None or not path.is_file():
        return [dict(p) for p in DEFAULT_PROMPTS]
    prompts = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = parse_prompt_line(line)
        if parsed is None:
            continue
        if parsed["prompt_id"] in seen:
            raise SystemExit(f"提示词 id 重复: {parsed['prompt_id']}")
        seen.add(parsed["prompt_id"])
        prompts.append(parsed)
    if not prompts:
        raise SystemExit(f"提示词文件为空: {path}")
    return prompts


def default_seed(prompt_id: str, base_seed: int) -> int:
    """确定性 seed：同一 prompt 在所有 checkpoint、所有重跑中同 seed。

    这是跨 epoch 指标曲线可比的前提。
    """
    offset = int(hashlib.sha256(prompt_id.encode("utf-8")).hexdigest()[:8], 16)
    return base_seed + offset % 10**6


def seeds_for(prompt_id: str, base_seed: int, samples_per_prompt: int) -> list[int]:
    """多张样例时的种子序列：seed + i*1000，第一张与单样例口径一致。"""
    seed = default_seed(prompt_id, base_seed)
    return [seed + i * 1000 for i in range(max(1, samples_per_prompt))]


def parse_epoch_selection(spec: str | None) -> set[int] | None:
    """解析 epoch 子集参数："all" / None -> 全部；"1-6,9" -> {1..6, 9}。"""
    if spec is None or spec.strip() == "all":
        return None
    selected: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            if not (lo.strip().isdigit() and hi.strip().isdigit()):
                raise SystemExit(f"epoch 选择格式不符（如 \"1-6,9\"）: {spec}")
            selected.update(range(int(lo), int(hi) + 1))
        elif part.isdigit():
            selected.add(int(part))
        else:
            raise SystemExit(f"epoch 选择格式不符（如 \"1-6,9\"）: {spec}")
    if not selected:
        raise SystemExit(f"epoch 选择为空: {spec}")
    return selected


def collect_checkpoints(
    experiment_dir: Path, epochs: str | None = None
) -> list[tuple[int, Path]]:
    """枚举 output/ 下带 epoch 后缀的 checkpoint，按 epoch 升序返回。

    无 epoch 后缀的最终版（如 nekoda_style_v1.safetensors）与最后一个
    epoch 内容重复，跳过。
    """
    output_dir = experiment_dir / "output"
    selected = parse_epoch_selection(epochs)
    found: list[tuple[int, Path]] = []
    if output_dir.is_dir():
        for path in sorted(output_dir.glob("*.safetensors")):
            match = EPOCH_PATTERN.search(path.name)
            if match is None:
                continue
            epoch = int(match.group(1))
            if selected is None or epoch in selected:
                found.append((epoch, path))
    if not found:
        hint = f"（选择: {epochs}）" if selected is not None else ""
        raise SystemExit(f"output/ 下没有可评估的 checkpoint{hint}: {output_dir}")
    return sorted(found)


def _find_toml_value(data: dict, key: str):
    """在 toml 顶层与各 section 中查找 key（kohya 配置两种写法都存在）。"""
    if key in data:
        return data[key]
    for value in data.values():
        if isinstance(value, dict):
            found = _find_toml_value(value, key)
            if found is not None:
                return found
    return None


def _read_config(experiment_dir: Path) -> dict:
    config_path = experiment_dir / "config.toml"
    if not config_path.is_file():
        raise SystemExit(f"实验缺少 config.toml: {experiment_dir}")
    return tomllib.loads(config_path.read_text(encoding="utf-8"))


def _resolve_repo_path(value: str) -> Path:
    """kohya 配置里的相对路径按仓库根解析（必须从仓库根运行训练）。"""
    path = Path(value)
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / path


def resolve_base_model(experiment_dir: Path) -> Path:
    """从 config.toml 读底模路径（pretrained_model_name_or_path）。"""
    value = _find_toml_value(_read_config(experiment_dir), "pretrained_model_name_or_path")
    if not value:
        raise SystemExit(f"config.toml 缺少 pretrained_model_name_or_path: {experiment_dir}")
    return _resolve_repo_path(str(value))


def infer_dataset_dir(experiment_dir: Path) -> Path:
    """从 config.toml 的 train_data_dir（指向 <dataset>/train）推断数据集目录。"""
    value = _find_toml_value(_read_config(experiment_dir), "train_data_dir")
    if not value:
        raise SystemExit(
            f"config.toml 缺少 train_data_dir，请用 --dataset 显式指定: {experiment_dir}"
        )
    return _resolve_repo_path(str(value)).parent


def load_split(
    dataset_dir: Path,
) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    """读 prepare 清单，按 split 列返回 ((image_id, 路径) 的 train / val 列表)。

    无 val 行时 val 为空列表，评估自动降级（只用训练集信号判定）。
    """
    manifest_path = dataset_dir / "manifest" / "prepare.parquet"
    if not manifest_path.is_file():
        raise SystemExit(
            f"缺少 prepare 清单: {manifest_path}\n（先运行 hello-lora prepare）"
        )
    manifest = pl.read_parquet(manifest_path)
    if "path" not in manifest.columns or "image_id" not in manifest.columns:
        raise SystemExit(f"prepare 清单缺少 path / image_id 列: {manifest_path}")
    train_pairs: list[tuple[str, Path]] = []
    val_pairs: list[tuple[str, Path]] = []
    split_col = manifest["split"].to_list() if "split" in manifest.columns else []
    for i, row in enumerate(manifest.iter_rows(named=True)):
        path = Path(row["path"])
        if not path.is_file():
            continue  # 清单里已不存在的文件直接跳过
        split = split_col[i] if i < len(split_col) else None
        (val_pairs if split == "val" else train_pairs).append((row["image_id"], path))
    if not train_pairs:
        raise SystemExit(f"prepare 清单为空（无可用参考图）: {manifest_path}")
    return train_pairs, val_pairs


def cos_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """L2 归一化后的余弦相似度矩阵 a @ b.T（纯 numpy）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a @ b.T


def sample_metrics(
    img_emb: np.ndarray,
    text_emb: np.ndarray,
    train_embs: np.ndarray,
    val_embs: np.ndarray | None,
) -> dict:
    """单张样例的 CLIP 指标。

    style_sim 取对参考集余弦的均值（画风保真看整体分布），
    nearest 取 max（复刻检测只看最像的那张）。
    """
    train_sims = (
        cos_sim_matrix(img_emb[None, :], train_embs)[0] if len(train_embs) else None
    )
    val_sims = (
        cos_sim_matrix(img_emb[None, :], val_embs)[0]
        if val_embs is not None and len(val_embs)
        else None
    )
    return {
        "clip_t": float(cos_sim_matrix(img_emb[None, :], text_emb[None, :])[0, 0]),
        "style_sim_train": float(train_sims.mean()) if train_sims is not None else None,
        "nearest_train_sim": float(train_sims.max()) if train_sims is not None else None,
        "style_sim_val": float(val_sims.mean()) if val_sims is not None else None,
    }


def diversity_of(embs: np.ndarray) -> float | None:
    """样例两两余弦相似度均值（越低越多样）；少于 2 张时无法计算。"""
    if len(embs) < 2:
        return None
    sims = cos_sim_matrix(embs, embs)
    n = len(embs)
    return float((sims.sum() - np.trace(sims)) / (n * (n - 1)))


def _mean_of(values: list) -> float | None:
    """均值；全部为 None（缺样例图）时返回 None 而不是让 numpy 报错。"""
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def _max_of(values: list) -> float | None:
    """最大值；全部为 None 时返回 None。"""
    valid = [v for v in values if v is not None]
    return float(np.max(valid)) if valid else None


def checkpoint_summary(
    checkpoint: str,
    epoch: int,
    rows: list[dict],
    base: dict,
    has_val: bool,
    diversity: float | None = None,
) -> dict:
    """单 checkpoint 汇总：各指标均值（nearest 取 max）+ 判定。

    diversity 由调用方算好传入（checkpoint 级指标，不属于单行）。
    verdict 规则（启发式，报告里注明）：
    - 欠拟合:      style_sim_train 相对 base 增益 < STYLE_GAIN_MIN
    - 过拟合风险:  记忆化超限 / 多样性塌缩 / 泛化 gap 相对 base 增大（需 val）
    - 其余:        正常
    """
    metrics = {
        "clip_t": _mean_of([r["clip_t"] for r in rows]),
        "style_sim_train": _mean_of([r["style_sim_train"] for r in rows]),
        "style_sim_val": _mean_of([r["style_sim_val"] for r in rows]) if has_val else None,
        "nearest_train_sim": _max_of([r["nearest_train_sim"] for r in rows]),
        "diversity": diversity,
    }
    summary = {"checkpoint": checkpoint, "epoch": epoch, "n_samples": len(rows), **metrics}

    if epoch == 0:  # base 基线自身不做判定
        summary["verdict"] = "基线"
        return summary
    if base.get("style_sim_train") is None or metrics["style_sim_train"] is None:
        summary["verdict"] = "样例不足"
        return summary

    if metrics["style_sim_train"] - base["style_sim_train"] < STYLE_GAIN_MIN:
        summary["verdict"] = "欠拟合"
        return summary

    reasons = []
    if metrics["nearest_train_sim"] is not None and metrics["nearest_train_sim"] > MEMORIZATION_MAX:
        reasons.append(f"记忆化({metrics['nearest_train_sim']:.3f})")
    if (
        base.get("diversity") is not None
        and metrics["diversity"] is not None
        and base["diversity"] - metrics["diversity"] > DIVERSITY_DROP
    ):
        reasons.append("多样性塌缩")
    if has_val and base.get("style_sim_val") is not None and metrics["style_sim_val"] is not None:
        gap = metrics["style_sim_train"] - metrics["style_sim_val"]
        base_gap = base["style_sim_train"] - base["style_sim_val"]
        if gap - base_gap > OVERFIT_GAP:
            reasons.append(f"泛化gap增大({gap - base_gap:+.3f})")
    summary["verdict"] = "过拟合风险" if reasons else "正常"
    if reasons:
        summary["verdict_reasons"] = reasons
    return summary


def pick_best_epoch(
    summaries: list[dict], base: dict, has_val: bool, n_val: int
) -> dict | None:
    """在候选 epoch 里选推荐 checkpoint。

    候选 = 非 base、非欠拟合、clip_t 未低于 base、多样性未塌缩；
    有足够 val 图时按 style_sim_val（泛化最优）取 argmax；
    无 val 时按 style_sim_train 取 argmax，并排除过拟合风险 epoch
    （否则会选中记忆化最重的那个）。
    """
    candidates = []
    for s in summaries:
        if s["epoch"] <= 0 or s["verdict"] == "欠拟合" or s["verdict"] == "样例不足":
            continue
        if s["clip_t"] is None or base.get("clip_t") is None:
            continue
        if s["clip_t"] < base["clip_t"] + PROMPT_CLIP_FLOOR:
            continue
        if (
            base.get("diversity") is not None
            and s.get("diversity") is not None
            and base["diversity"] - s["diversity"] > DIVERSITY_DROP
        ):
            continue
        if not has_val and s["verdict"] == "过拟合风险":
            continue
        candidates.append(s)
    if not candidates:
        return None
    if has_val and n_val >= MIN_VAL_IMAGES:
        key = "style_sim_val"
    else:
        key = "style_sim_train"
    return max(
        (c for c in candidates if c.get(key) is not None),
        key=lambda s: s[key],
        default=None,
    )


def _load_clip(model_dir: Path, device: str, dtype: str):
    """加载 CLIP 模型与预处理（需要 inference 依赖组）。"""
    try:
        import torch
        from transformers import CLIPImageProcessor, CLIPTokenizer, CLIPModel
    except ImportError as exc:
        raise SystemExit(
            "缺少推理依赖（torch / transformers），请先运行: uv sync --group inference\n"
            f"（{exc}）"
        ) from exc

    torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype]
    model = CLIPModel.from_pretrained(model_dir, torch_dtype=torch_dtype)
    model.to(device).eval()
    processor = CLIPImageProcessor.from_pretrained(model_dir)
    tokenizer = CLIPTokenizer.from_pretrained(model_dir)
    return model, processor, tokenizer


def _feat_tensor(feats):
    """取 get_*_features 的 embedding 张量。

    transformers v5 起返回 BaseModelOutputWithPooling 对象，投影后的
    embedding 在 pooler_output；旧版直接返回 Tensor，两种都兼容。
    """
    import torch

    return feats if torch.is_tensor(feats) else feats.pooler_output


def _encode_images(
    model, processor, paths: list[Path], device: str, batch_size: int = CLIP_BATCH_SIZE
) -> np.ndarray:
    """批量计算图像 embedding（float32 numpy，未归一化）。"""
    import torch

    outputs = []
    with torch.inference_mode():
        for i in range(0, len(paths), batch_size):
            images = [Image.open(p).convert("RGB") for p in paths[i : i + batch_size]]
            inputs = processor(images=images, return_tensors="pt").to(device)
            feats = _feat_tensor(model.get_image_features(**inputs))
            outputs.append(feats.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _encode_texts(model, tokenizer, texts: list[str], device: str) -> np.ndarray:
    """计算文本 embedding（float32 numpy，未归一化）。"""
    import torch

    with torch.inference_mode():
        inputs = tokenizer(text=texts, padding=True, truncation=True, return_tensors="pt").to(device)
        feats = _feat_tensor(model.get_text_features(**inputs))
    return feats.float().cpu().numpy()


def _load_ref_cache(cache_path: Path, repo: str) -> dict[str, np.ndarray]:
    """读参考图 embedding 缓存（换 CLIP 模型后自动失效重建）。"""
    if not cache_path.is_file():
        return {}
    data = np.load(cache_path)
    if str(data["model"][0]) != repo:
        return {}
    return {
        str(image_id): np.asarray(emb)
        for image_id, emb in zip(data["ids"], data["emb"])
    }


def _save_ref_cache(
    cache_path: Path, repo: str, mapping: dict[str, np.ndarray]
) -> None:
    image_ids = sorted(mapping)
    np.savez(
        cache_path,
        model=np.array([repo]),
        ids=np.array(image_ids),
        emb=np.stack([mapping[i] for i in image_ids]).astype(np.float32),
    )


def _reference_embs(
    pairs: list[tuple[str, Path]],
    cache: dict[str, np.ndarray],
    model,
    processor,
    device: str,
) -> list[tuple[str, np.ndarray | None]]:
    """返回 (image_id, embedding | None) 列表，缺失的现场编码并写回缓存。"""
    todo = [(i, p) for i, p in pairs if i not in cache]
    if todo:
        emb_matrix = _encode_images(model, processor, [p for _, p in todo], device)
        for (image_id, _), emb in zip(todo, emb_matrix):
            cache[image_id] = emb
    return [(i, cache.get(i)) for i, _ in pairs]


def evaluate_experiment(
    experiment_dir: Path,
    dataset_dir: Path,
    epochs: str | None = None,
    prompts_path: Path | None = None,
    base_seed: int = DEFAULT_BASE_SEED,
    samples_per_prompt: int = 1,
    clip_alias: str = DEFAULT_CLIP,
    resolution: int = DEFAULT_RESOLUTION,
    steps: int = DEFAULT_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    device: str = "cuda",
    dtype: str = "fp16",
    offload: str = "sequential",
    generate_samples: bool = True,
    force: bool = False,
) -> tuple[pl.DataFrame, dict]:
    """评估实验目录下的各 checkpoint，返回 (eval manifest, 报告 dict)。

    流程：固定 prompt × 确定性 seed 生成样例（底模只加载一次，checkpoint
    间只换 LoRA 权重）-> CLIP 编码样例与参考图（参考图 embedding 缓存）
    -> 逐 checkpoint 汇总判定 -> 写 eval.parquet 与 eval.report.json。
    """
    if clip_alias not in CLIP_REPOS:
        raise SystemExit(f"未知 CLIP 模型别名: {clip_alias}（可选 {', '.join(CLIP_REPOS)}）")
    repo = CLIP_REPOS[clip_alias]

    prompts = load_prompts(prompts_path if prompts_path else experiment_dir / "prompts.txt")
    checkpoints = collect_checkpoints(experiment_dir, epochs)
    base_model = resolve_base_model(experiment_dir)
    if not base_model.is_file():
        raise SystemExit(f"底模文件不存在: {base_model}")
    train_pairs, val_pairs = load_split(dataset_dir)

    samples_dir = experiment_dir / SAMPLES_DIR_NAME
    manifest_dir = experiment_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # ---- 样例图：先生成 base 基线，再逐 checkpoint 换 LoRA ----
    if generate_samples:
        try:
            from hello_lora.generate import (
                build_pipeline,
                encode_prompts,
                generate_for_checkpoint,
                stage_unet,
            )
        except ImportError as exc:
            raise SystemExit(
                "缺少推理依赖（torch / diffusers），请先运行: uv sync --group inference\n"
                f"（{exc}）"
            ) from exc

        pipe = build_pipeline(base_model, device=device, dtype=dtype, offload=offload)
        # prompt 只编码一次（与 checkpoint 无关），编码完腾出文本编码器显存
        embeds_map, negative = encode_prompts(
            pipe, [p["prompt"] for p in prompts], device=device, offload=offload
        )
        stage_unet(pipe, device=device, offload=offload)
        for lora, name in [(None, "base")] + [(p, p.stem) for _, p in checkpoints]:
            jobs = [
                (
                    samples_dir / name / f"{p['prompt_id']}_seed{s}.png",
                    p["prompt"],
                    s,
                )
                for p in prompts
                for s in seeds_for(p["prompt_id"], base_seed, samples_per_prompt)
            ]
            generate_for_checkpoint(
                pipe, lora, jobs, embeds=embeds_map, negative=negative,
                resolution=resolution, steps=steps,
                guidance=guidance, force=force, device=device,
            )

    # ---- CLIP 打分 ----
    # 依赖检查放在模型下载之前：没装 inference 组时先给出明确指引，不白等下载
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "缺少推理依赖（torch / transformers），请先运行: uv sync --group inference\n"
            f"（{exc}）"
        ) from exc
    model_dir = ensure_clip_model(default_model_dir(), repo)
    model, processor, tokenizer = _load_clip(model_dir, device, dtype)
    print(f"CLIP 打分模型: {repo} ({model_dir})")

    text_embs = _encode_texts(
        model, tokenizer, [p["prompt"] for p in prompts], device
    )
    prompt_by_id = {p["prompt_id"]: (i, p) for i, p in enumerate(prompts)}

    cache_path = manifest_dir / "clip_ref_cache.npz"
    cache = _load_ref_cache(cache_path, repo)
    train_ref = _reference_embs(train_pairs, cache, model, processor, device)
    val_ref = _reference_embs(val_pairs, cache, model, processor, device) if val_pairs else []
    _save_ref_cache(cache_path, repo, cache)

    train_embs = np.stack([e for _, e in train_ref if e is not None])
    val_embs = np.stack([e for _, e in val_ref if e is not None]) if val_ref else None
    has_val = val_embs is not None and len(val_embs) > 0

    # ---- 组装逐样例行 ----
    units: list[tuple[str, int]] = [("base", 0)] + [(p.stem, epoch) for epoch, p in checkpoints]
    rows: dict[str, list[dict]] = {name: [] for name, _ in units}
    unit_embs: dict[str, list[np.ndarray]] = {name: [] for name, _ in units}
    for name, epoch in units:
        for p in prompts:
            idx, _ = prompt_by_id[p["prompt_id"]]
            for s in seeds_for(p["prompt_id"], base_seed, samples_per_prompt):
                image = samples_dir / name / f"{p['prompt_id']}_seed{s}.png"
                row = {
                    "checkpoint": name,
                    "epoch": epoch,
                    "prompt_id": p["prompt_id"],
                    "prompt": p["prompt"],
                    "kind": p["kind"],
                    "seed": s,
                    "image": image.relative_to(experiment_dir).as_posix(),
                    "clip_t": None,
                    "style_sim_train": None,
                    "style_sim_val": None,
                    "nearest_train_sim": None,
                    "notes": "",
                }
                if not image.is_file():
                    row["notes"] = "缺少样例图"
                    rows[name].append(row)
                    continue
                with Image.open(image) as im:
                    img_emb = _encode_images(model, processor, [image], device)[0]
                metrics = sample_metrics(img_emb, text_embs[idx], train_embs, val_embs)
                row.update(metrics)
                rows[name].append(row)
                unit_embs[name].append(img_emb)

    # ---- 汇总 / 判定 / 报告 ----
    def unit_diversity(name: str) -> float | None:
        return diversity_of(np.stack(unit_embs[name])) if unit_embs[name] else None

    base_summary = checkpoint_summary(
        "base", 0, rows["base"], {}, has_val, unit_diversity("base")
    )
    if base_summary["style_sim_train"] is None:
        raise SystemExit(
            "base 基线样例缺失，无法判定拟合程度（去掉 --skip-generation 先生成基线）"
        )

    summaries = [base_summary]
    for name, epoch in units:
        if name == "base":
            continue
        summaries.append(
            checkpoint_summary(
                name, epoch, rows[name], base_summary, has_val, unit_diversity(name)
            )
        )

    best = pick_best_epoch(summaries, base_summary, has_val, len(val_pairs))
    manifest = pl.DataFrame([r for name, _ in units for r in rows[name]], schema=EVAL_SCHEMA)

    manifest_path = manifest_dir / "eval.parquet"
    manifest.write_parquet(manifest_path)

    report = {
        "experiment": experiment_dir.name,
        "dataset": str(dataset_dir),
        "clip_model": repo,
        "prompts": prompts,
        "n_train_images": len(train_pairs),
        "n_val_images": len(val_pairs),
        "has_val": has_val,
        "thresholds": {
            "style_gain_min": STYLE_GAIN_MIN,
            "memorization_max": MEMORIZATION_MAX,
            "diversity_drop": DIVERSITY_DROP,
            "overfit_gap": OVERFIT_GAP,
            "note": "均为经验规则（启发式），仅供排序参考",
        },
        "base": base_summary,
        "per_checkpoint": summaries,
        "best": best,
        "notes": []
        if has_val
        else [
            f"无验证集，过拟合判断仅基于记忆化 / 多样性 / clip_t 指标；"
            f"建议下次 prepare 时加 --val-ratio 0.1（至少 {MIN_VAL_IMAGES} 张 val 图更可信）"
        ],
    }
    report_path = manifest_dir / "eval.report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"评估清单已写入: {manifest_path} ({manifest.height} 行)")
    print(f"评估报告已写入: {report_path}")
    return manifest, report


def summarize_report(report: dict) -> str:
    """人读摘要：逐 epoch 指标表 + 推荐结论（CLI 直接 print）。"""
    lines = []
    if not report["has_val"]:
        lines += report["notes"]

    header = (
        f"{'epoch':>6}  {'style_sim':>9}  {'clip_t':>7}  "
        f"{'diversity':>9}  {'nearest':>7}  verdict"
    )
    lines.append(header)
    for s in report["per_checkpoint"]:
        def fmt(v, digits=4):
            return f"{v:.{digits}f}" if isinstance(v, float) else "-"
        label = "base" if s["epoch"] == 0 else str(s["epoch"])
        lines.append(
            f"{label:>6}  {fmt(s['style_sim_train'])}  {fmt(s['clip_t'])}  "
            f"{fmt(s['diversity'])}  {fmt(s['nearest_train_sim'])}  {s['verdict']}"
        )

    best = report["best"]
    if best:
        basis = (
            "style_sim_val（泛化最优）"
            if report["has_val"] and report["n_val_images"] >= MIN_VAL_IMAGES
            else "style_sim_train（无验证集，注意结合样例图人工复核）"
        )
        lines.append(f"推荐 epoch: {best['epoch']}（{basis}）")
    else:
        lines.append("没有满足候选条件的 epoch，请检查样例图后人工判断")
    return "\n".join(lines)
