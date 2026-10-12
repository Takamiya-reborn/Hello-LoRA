"""eval 模块测试：纯函数直测 + monkeypatch 假件的编排冒烟，不下载模型不跑推理。"""

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PIL import Image

from hello_lora import eval as eval_mod
from hello_lora.eval import (
    EVAL_SCHEMA,
    checkpoint_summary,
    collect_checkpoints,
    cos_sim_matrix,
    default_seed,
    diversity_of,
    infer_dataset_dir,
    load_prompts,
    parse_prompt_line,
    pick_best_epoch,
    sample_metrics,
    seeds_for,
)


def _emb(seed_text: str, dim: int = 8) -> np.ndarray:
    """由文本确定性派生的假 embedding，供冒烟测试用。"""
    rng = np.random.default_rng(
        int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)
    )
    return rng.standard_normal(dim)


# ---- 提示词解析 ----


def test_parse_prompt_line() -> None:
    assert parse_prompt_line("p01|style|1girl, solo") == {
        "prompt_id": "p01",
        "kind": "style",
        "prompt": "1girl, solo",
    }
    # 空行与注释返回 None；两侧空白被清理
    assert parse_prompt_line("   ") is None
    assert parse_prompt_line("# comment") is None
    parsed = parse_prompt_line("  p2 | unseen | cat, rain ")
    assert parsed == {"prompt_id": "p2", "kind": "unseen", "prompt": "cat, rain"}


def test_parse_prompt_line_errors() -> None:
    with pytest.raises(SystemExit):  # 缺 kind
        parse_prompt_line("p01|1girl, solo")
    with pytest.raises(SystemExit):  # 未知 kind
        parse_prompt_line("p01|other|1girl")
    with pytest.raises(SystemExit):  # 空 prompt
        parse_prompt_line("p01|style|")


def test_load_prompts_default(tmp_path: Path) -> None:
    prompts = load_prompts(None)
    assert len(prompts) >= 5
    assert {p["kind"] for p in prompts} <= {"style", "unseen"}
    assert len({p["prompt_id"] for p in prompts}) == len(prompts)

    # 文件存在则按行解析，空文件报错
    path = tmp_path / "prompts.txt"
    path.write_text("# 注释\np01|style|a, b\n", encoding="utf-8")
    assert load_prompts(path) == [{"prompt_id": "p01", "kind": "style", "prompt": "a, b"}]
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_prompts(empty)


# ---- seed 派生 ----


def test_default_seed_deterministic() -> None:
    assert default_seed("p01", 42) == default_seed("p01", 42)
    assert default_seed("p01", 42) != default_seed("p02", 42)
    assert default_seed("p01", 42) != default_seed("p01", 43)
    assert seeds_for("p01", 42, 3)[0] == default_seed("p01", 42)
    assert seeds_for("p01", 42, 3) == [default_seed("p01", 42), default_seed("p01", 42) + 1000, default_seed("p01", 42) + 2000]


# ---- checkpoint 枚举 ----


def test_collect_checkpoints(tmp_path: Path) -> None:
    out = tmp_path / "output"
    out.mkdir()
    for epoch in (1, 2, 3, 7):
        (out / f"mymodel-{epoch:06d}.safetensors").write_bytes(b"x")
    (out / "mymodel.safetensors").write_bytes(b"x")  # final 版应被跳过
    (out / "ignore.txt").write_text("x")

    found = collect_checkpoints(tmp_path)
    assert found == [
        (1, out / "mymodel-000001.safetensors"),
        (2, out / "mymodel-000002.safetensors"),
        (3, out / "mymodel-000003.safetensors"),
        (7, out / "mymodel-000007.safetensors"),
    ]
    # 子集选择
    subset = collect_checkpoints(tmp_path, "1-3")
    assert [e for e, _ in subset] == [1, 2, 3]


def test_collect_checkpoints_empty(tmp_path: Path) -> None:
    (tmp_path / "output").mkdir()
    with pytest.raises(SystemExit):
        collect_checkpoints(tmp_path)


def test_parse_epoch_selection_invalid() -> None:
    with pytest.raises(SystemExit):
        eval_mod.parse_epoch_selection("abc")
    with pytest.raises(SystemExit):
        eval_mod.parse_epoch_selection("")


# ---- config.toml 解析 ----


def test_infer_dataset_dir(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        'pretrained_model_name_or_path = "assets/models/ckpt/base.safetensors"\n'
        'train_data_dir = "assets/data/some_ds/train"\n',
        encoding="utf-8",
    )
    dataset = infer_dataset_dir(tmp_path)
    assert dataset.name == "some_ds"
    assert dataset.name != "train"


# ---- 指标纯函数 ----


def test_cos_sim_matrix() -> None:
    a = np.array([[1.0, 0.0], [0.0, 2.0]])
    b = np.array([[2.0, 0.0], [0.0, -1.0]])
    sims = cos_sim_matrix(a, b)
    assert np.allclose(sims, [[1.0, 0.0], [0.0, -1.0]])
    self_sims = cos_sim_matrix(a, a)
    assert np.allclose(np.diag(self_sims), 1.0)


def test_sample_metrics() -> None:
    img = np.array([1.0, 0.0, 0.0])
    text = np.array([1.0, 1.0, 0.0])
    train = np.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [-1.0, 0.0, 0.0]])
    val = np.array([[0.0, 1.0, 0.0]])

    m = sample_metrics(img, text, train, val)
    # clip_t = cos([1,0,0],[1,1,0]) / sqrt(2)
    assert m["clip_t"] == pytest.approx(1 / np.sqrt(2))
    assert m["style_sim_train"] == pytest.approx(np.mean([1.0, 0.6, -1.0]))
    assert m["nearest_train_sim"] == pytest.approx(1.0)  # max，不是均值
    assert m["style_sim_val"] == pytest.approx(0.0)

    # 无 val 时 style_sim_val 为 None
    m_no_val = sample_metrics(img, text, train, None)
    assert m_no_val["style_sim_val"] is None


def test_diversity_of() -> None:
    assert diversity_of(np.array([[1.0, 0.0]])) is None  # 单张无法计算
    embs = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    # 两两余弦: [0, -1, 0] -> 均值 -1/3
    assert diversity_of(embs) == pytest.approx(-1 / 3)


# ---- 汇总与判定 ----


def make_rows(clip_t: float, style: float, nearest: float, n: int = 2) -> list[dict]:
    return [
        {
            "clip_t": clip_t,
            "style_sim_train": style,
            "style_sim_val": None,
            "nearest_train_sim": nearest,
        }
        for _ in range(n)
    ]


def base_summary(style: float, clip_t: float = 0.30, diversity: float = 0.60) -> dict:
    return {
        "checkpoint": "base",
        "epoch": 0,
        "clip_t": clip_t,
        "style_sim_train": style,
        "style_sim_val": None,
        "nearest_train_sim": 0.5,
        "diversity": diversity,
    }


def test_checkpoint_summary_underfit() -> None:
    base = base_summary(style=0.60)
    # style 增益 < 0.02 -> 欠拟合
    s = checkpoint_summary("m-000001", 1, make_rows(0.32, 0.605, 0.7), base, False)
    assert s["verdict"] == "欠拟合"


def test_checkpoint_summary_overfit_memorization() -> None:
    base = base_summary(style=0.60)
    s = checkpoint_summary("m-000009", 9, make_rows(0.35, 0.75, 0.95), base, False)
    assert s["verdict"] == "过拟合风险"
    assert any("记忆化" in r for r in s["verdict_reasons"])


def test_checkpoint_summary_normal() -> None:
    base = base_summary(style=0.60)
    s = checkpoint_summary(
        "m-000005", 5, make_rows(0.35, 0.70, 0.70), base, False, diversity=0.58
    )
    assert s["verdict"] == "正常"


def test_checkpoint_summary_diversity_collapse() -> None:
    base = base_summary(style=0.60, diversity=0.60)
    s = checkpoint_summary(
        "m-000012", 12, make_rows(0.35, 0.70, 0.70), base, False, diversity=0.50
    )
    assert s["verdict"] == "过拟合风险"
    assert any("多样性" in r for r in s["verdict_reasons"])


def test_checkpoint_summary_val_gap() -> None:
    base = base_summary(style=0.60)
    base["style_sim_val"] = 0.55  # base gap = 0.05
    rows = make_rows(0.35, 0.70, 0.70)
    for r in rows:
        r["style_sim_val"] = 0.50  # gap = 0.20，增大 0.15 > 0.03
    s = checkpoint_summary("m-000012", 12, rows, base, True, diversity=0.58)
    assert s["verdict"] == "过拟合风险"
    assert any("泛化" in r for r in s["verdict_reasons"])


def make_summary(epoch: int, style: float, clip_t: float = 0.32, **extra) -> dict:
    s = {
        "checkpoint": f"m-{epoch:06d}",
        "epoch": epoch,
        "clip_t": clip_t,
        "style_sim_train": style,
        "style_sim_val": None,
        "nearest_train_sim": 0.7,
        "diversity": 0.58,
        "verdict": "正常",
    }
    s.update(extra)
    return s


def test_pick_best_epoch_no_val() -> None:
    base = base_summary(style=0.60)
    summaries = [
        make_summary(1, 0.62),
        make_summary(5, 0.70),
        make_summary(9, 0.78, verdict="过拟合风险"),  # 无 val 时排除过拟合风险
        make_summary(12, 0.68),
    ]
    best = pick_best_epoch(summaries, base, has_val=False, n_val=0)
    assert best is not None and best["epoch"] == 5

    # clip_t 低于 base 的 epoch 被排除
    low_clip = [make_summary(1, 0.70, clip_t=0.25)]
    assert pick_best_epoch(low_clip, base, has_val=False, n_val=0) is None


def test_pick_best_epoch_with_val() -> None:
    base = base_summary(style=0.60)
    base["style_sim_val"] = 0.55
    summaries = [
        make_summary(3, 0.68, style_sim_val=0.66),
        make_summary(6, 0.75, style_sim_val=0.70),  # val 相似度最高 -> 推荐
        make_summary(9, 0.80, style_sim_val=0.60, verdict="过拟合风险"),
    ]
    best = pick_best_epoch(summaries, base, has_val=True, n_val=10)
    assert best is not None and best["epoch"] == 6


# ---- 编排冒烟：假件替换生成与 CLIP ----


@pytest.fixture
def mini_experiment(tmp_path: Path) -> tuple[Path, Path]:
    """迷你实验 + 迷你数据集：2 张 train 图 + 2 张 val 图，2 个假 checkpoint。"""
    dataset = tmp_path / "mini_ds"
    dataset.mkdir()
    rows = []
    for i in range(4):
        img = dataset / f"img{i}.jpg"
        Image.new("RGB", (16, 16), (i * 40, 100, 100)).save(img)
        rows.append(
            {
                "image_id": hashlib.sha256(img.read_bytes()).hexdigest()[:16],
                "path": str(img),
                "split": "train" if i < 2 else "val",
            }
        )
    manifest_dir = dataset / "manifest"
    manifest_dir.mkdir()
    pl.DataFrame(rows, schema={"image_id": pl.String, "path": pl.String, "split": pl.String}).write_parquet(
        manifest_dir / "prepare.parquet"
    )

    experiment = tmp_path / "mini_v1"
    (experiment / "output").mkdir(parents=True)
    base_model = tmp_path / "base.safetensors"
    base_model.write_bytes(b"fake")
    (experiment / "config.toml").write_text(
        f'pretrained_model_name_or_path = "{base_model.as_posix()}"\n'
        f'train_data_dir = "{(dataset / "train").as_posix()}"\n',
        encoding="utf-8",
    )
    (experiment / "output" / "mini_v1-000001.safetensors").write_bytes(b"x")
    (experiment / "output" / "mini_v1-000002.safetensors").write_bytes(b"x")
    return experiment, dataset


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    """用假模块 / 假函数替换生成与 CLIP 编码，主环境无 torch 也能跑编排。"""
    fake_generate = types.ModuleType("hello_lora.generate")

    def fake_build_pipeline(base_model, device="cuda", dtype="fp16", offload="model"):
        return object()

    def fake_encode_prompts(pipe, prompts, device="cuda", offload="model"):
        embeds_map = {text: (object(), object()) for text in set(prompts)}
        return embeds_map, (object(), object())

    def fake_stage_unet(pipe, device="cuda", offload="model"):
        return None

    def fake_generate_for_checkpoint(
        pipe, lora_path, jobs, embeds=None, negative=None,
        resolution=0, steps=0, guidance=0.0, force=False, device="cuda",
    ):
        assert all(prompt in embeds for _, prompt, _ in jobs), "prompt 必须已预编码"
        for out, prompt, seed in jobs:
            out.parent.mkdir(parents=True, exist_ok=True)
            # 用 prompt_id + seed 决定颜色，embedding 才能对样例做出区分
            Image.new("RGB", (8, 8), (seed % 255, len(prompt), 128)).save(out)
        return [out for out, _, _ in jobs]

    fake_generate.build_pipeline = fake_build_pipeline
    fake_generate.encode_prompts = fake_encode_prompts
    fake_generate.stage_unet = fake_stage_unet
    fake_generate.generate_for_checkpoint = fake_generate_for_checkpoint
    monkeypatch.setitem(sys.modules, "hello_lora.generate", fake_generate)

    monkeypatch.setattr(eval_mod, "ensure_clip_model", lambda root, repo: Path("fake-clip"))
    monkeypatch.setattr(eval_mod, "_load_clip", lambda model_dir, device, dtype: (None, None, None))
    monkeypatch.setattr(
        eval_mod,
        "_encode_images",
        lambda model, processor, paths, device, batch_size=16: np.stack(
            [_emb(f"img:{p.name}") for p in paths]
        ),
    )
    monkeypatch.setattr(
        eval_mod,
        "_encode_texts",
        lambda model, tokenizer, texts, device: np.stack([_emb(f"txt:{t}") for t in texts]),
    )


def test_evaluate_experiment_smoke(mini_experiment: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment, dataset = mini_experiment
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("p01|style|test prompt one\np02|unseen|test prompt two\n", encoding="utf-8")
    _install_fakes(monkeypatch)

    manifest, report = eval_mod.evaluate_experiment(
        experiment, dataset, prompts_path=prompts, generate_samples=True
    )

    # (base + 2 checkpoint) × 2 prompt × 1 seed = 6 行，列与 schema 一致
    assert manifest.height == 6
    assert manifest.columns == list(EVAL_SCHEMA)
    # 样例图落盘：base 与每个 checkpoint 各 2 张
    samples = experiment / "samples" / "eval"
    seed = default_seed("p01", 42)
    assert (samples / "base" / f"p01_seed{seed}.png").is_file()
    assert len(list((samples / "mini_v1-000002").glob("*.png"))) == 2

    report_path = experiment / "manifest" / "eval.report.json"
    assert report_path.is_file()
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["best"] is None or "epoch" in loaded["best"]
    assert [s["epoch"] for s in loaded["per_checkpoint"]] == [0, 1, 2]
    assert (experiment / "manifest" / "eval.parquet").is_file()
    # 参考图 embedding 缓存已写入
    assert (experiment / "manifest" / "clip_ref_cache.npz").is_file()
    # 行内指标已填充（假 embedding 下 clip_t 是确定值）
    assert manifest.filter(pl.col("checkpoint") == "base")["clip_t"][0] is not None


def test_evaluate_experiment_skip_generation(mini_experiment: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--skip-generation：复用现有样例图重打分；完全无样例时给出明确报错。"""
    experiment, dataset = mini_experiment
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("p01|style|test prompt one\n", encoding="utf-8")
    _install_fakes(monkeypatch)

    # 先跑一次生成样例，再跳过生成重打分，结果行数一致
    first, _ = eval_mod.evaluate_experiment(
        experiment, dataset, prompts_path=prompts, generate_samples=True
    )
    second, _ = eval_mod.evaluate_experiment(
        experiment, dataset, prompts_path=prompts, generate_samples=False
    )
    assert second.height == first.height

    # 删掉 base 样例后跳过生成 -> 无法判定拟合，报错
    for png in (experiment / "samples" / "eval" / "base").glob("*.png"):
        png.unlink()
    with pytest.raises(SystemExit):
        eval_mod.evaluate_experiment(
            experiment, dataset, prompts_path=prompts, generate_samples=False
        )


def test_summarize_report_formatting() -> None:
    report = {
        "has_val": False,
        "notes": ["无验证集提示"],
        "n_val_images": 0,
        "per_checkpoint": [
            {
                "epoch": 0,
                "style_sim_train": 0.6,
                "clip_t": 0.3,
                "diversity": 0.6,
                "nearest_train_sim": 0.5,
                "verdict": "基线",
            },
            {
                "epoch": 1,
                "style_sim_train": 0.7,
                "clip_t": 0.32,
                "diversity": 0.55,
                "nearest_train_sim": 0.7,
                "verdict": "正常",
            },
        ],
        "best": {"epoch": 1},
    }
    text = eval_mod.summarize_report(report)
    assert "无验证集提示" in text
    assert "base" in text and "推荐 epoch: 1" in text
