"""Hello-LoRA 命令行入口。

每个数据处理步骤对应一个子命令。数据集参数只需填 <dataset> 名字，
固定解析到 assets/data/<dataset>（传入已存在的目录路径时原样使用）；
涉及随机性的步骤额外支持 --seed。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from hello_lora.eval import (
    CLIP_REPOS,
    DEFAULT_BASE_SEED,
    DEFAULT_CLIP,
    DEFAULT_GUIDANCE,
    DEFAULT_RESOLUTION,
    DEFAULT_STEPS,
    evaluate_experiment,
    infer_dataset_dir,
    summarize_report,
)
from hello_lora.prepare import IMAGE_EXTS, prepare_dataset, summarize_dataset
from hello_lora.profiles import get as profile_get, resolve_profile
from hello_lora.scan import scan_directory, summarize
from hello_lora.tag import (
    BATCH_SIZE,
    CHARACTER_THRESHOLD,
    DEFAULT_MODEL,
    DEFAULT_PURPOSE,
    GENERAL_THRESHOLD,
    MODEL_REPOS,
    PURPOSES,
    tag_dataset,
)

ASSETS_DATA = Path("assets/data")
ASSETS_EXPERIMENTS = Path("assets/experiments")


def resolve_dataset_dir(name: str) -> Path:
    """<dataset> 名字 -> assets/data/<name>；已是存在的目录则原样使用。"""
    path = Path(name)
    if path.is_dir():
        return path
    return ASSETS_DATA / name


def cmd_scan(args: argparse.Namespace) -> None:
    input_dir = resolve_dataset_dir(args.dataset)
    if not input_dir.is_dir():
        raise SystemExit(f"数据集目录不存在: {input_dir}")

    output = input_dir / "manifest" / "scan"
    manifest_path = output.with_suffix(".parquet")
    # 增量扫描：上次清单存在时复用未变化文件的记录（size + mtime 校验）
    previous = None
    if manifest_path.exists() and not args.no_cache:
        previous = pl.read_parquet(manifest_path)
    manifest = scan_directory(input_dir, previous)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(manifest_path)
    report = summarize(manifest)
    report_path = output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"清单已写入: {manifest_path} ({manifest.height} 个文件)")
    print(f"扫描报告已写入: {report_path}")
    print(
        f"待人工确认: {len(report['needs_review'])} 个；"
        f"完全重复组: {report['duplicate_exact_groups']}；"
        f"近重复组: {len(report['duplicate_near_groups'])}"
    )


def cmd_tag(args: argparse.Namespace) -> None:
    dataset_dir = resolve_dataset_dir(args.dataset)
    if not dataset_dir.is_dir():
        raise SystemExit(f"数据集目录不存在: {dataset_dir}")
    _, profile = resolve_profile(args.profile)
    # --model 接受内置别名或任意 HF repo id（含 / 时视为 repo id）
    repo = MODEL_REPOS.get(args.model, args.model)
    manifest, skipped = tag_dataset(
        dataset_dir,
        repo=repo,
        purpose=args.purpose,
        general_threshold=args.threshold,
        character_threshold=args.character_threshold,
        batch_size=args.batch_size or profile_get(profile, "tag.batch_size", BATCH_SIZE),
        overwrite=args.overwrite,
    )
    if skipped:
        print(f"跳过 {len(skipped)} 个文件:")
        for name, reason in skipped[:10]:
            print(f"  {name}: {reason}")
        if len(skipped) > 10:
            print(f"  ... 以及另外 {len(skipped) - 10} 个文件")
    print(f"打标清单已写入: {dataset_dir / 'manifest' / 'tag.parquet'} ({manifest.height} 行)")
    if manifest.height:
        dist = manifest.group_by("rating").len().sort("len", descending=True)
        print("rating 分布:", ", ".join(f"{r['rating']}={r['len']}" for r in dist.to_dicts()))


def cmd_prepare(args: argparse.Namespace) -> None:
    dataset_dir = resolve_dataset_dir(args.dataset)
    manifest, train_dir, skipped = prepare_dataset(
        dataset_dir,
        repeats=args.repeats,
        ratings=args.ratings,
        trigger_word=args.trigger_word,
        keep_artist_tag=args.keep_artist_tag,
        val_ratio=args.val_ratio,
    )
    if skipped:
        print(f"跳过 {len(skipped)} 个文件:")
        for name, reason in skipped[:10]:
            print(f"  {name}: {reason}")
        if len(skipped) > 10:
            print(f"  ... 以及另外 {len(skipped) - 10} 个文件")
    print(f"manifest 已写入: {dataset_dir / 'manifest' / 'prepare.parquet'} ({manifest.height} 行)")
    n_images = sum(
        1 for p in train_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    print(f"训练目录: {train_dir} -> {n_images} 张图")
    n_val = manifest.filter(pl.col("split") == "val").height
    if n_val:
        print(f"验证目录: {dataset_dir / 'val'} -> {n_val} 张图（不参与训练，供评估用）")
    print(summarize_dataset(manifest))


def cmd_eval(args: argparse.Namespace) -> None:
    # 实验参数语义与 resolve_dataset_dir 一致：名字 -> assets/experiments/<id>
    experiment_dir = Path(args.experiment)
    if not experiment_dir.is_dir():
        experiment_dir = ASSETS_EXPERIMENTS / args.experiment
    if not experiment_dir.is_dir():
        raise SystemExit(f"实验目录不存在: {ASSETS_EXPERIMENTS / args.experiment}")
    _, profile = resolve_profile(args.profile)
    # CLI flag > profile > 代码内置默认
    clip_alias = args.clip_model or profile_get(profile, "eval.clip_model", DEFAULT_CLIP)
    resolution = args.resolution or profile_get(profile, "eval.resolution", DEFAULT_RESOLUTION)
    offload = args.offload or profile_get(profile, "eval.offload", "sequential")
    dataset_dir = (
        resolve_dataset_dir(args.dataset)
        if args.dataset
        else infer_dataset_dir(experiment_dir)
    )
    if not dataset_dir.is_dir():
        raise SystemExit(f"数据集目录不存在: {dataset_dir}")
    prompts_path = Path(args.prompts) if args.prompts else None
    _, report = evaluate_experiment(
        experiment_dir,
        dataset_dir,
        epochs=args.epochs,
        prompts_path=prompts_path,
        base_seed=args.base_seed,
        samples_per_prompt=args.samples_per_prompt,
        clip_alias=clip_alias,
        resolution=resolution,
        steps=args.steps,
        guidance=args.guidance,
        device=args.device,
        dtype=args.dtype,
        offload=offload,
        generate_samples=not args.skip_generation,
        force=args.force,
    )
    print(summarize_report(report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hello-lora", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    scan_parser = sub.add_parser(
        "scan", help="扫描数据集的图片，生成 manifest 清单和扫描报告"
    )
    scan_parser.add_argument(
        "dataset", help="数据集名（解析为 assets/data/<dataset>）或目录路径"
    )
    scan_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="忽略上次清单，强制全量重扫（默认对未变化的文件增量复用）",
    )
    scan_parser.set_defaults(func=cmd_scan)

    tag_parser = sub.add_parser(
        "tag", help="用 WD14 模型给图片打标，生成同名 .txt 标签文件"
    )
    tag_parser.add_argument(
        "dataset", help="数据集名（解析为 assets/data/<dataset>）或目录路径"
    )
    tag_parser.add_argument(
        "--purpose",
        choices=PURPOSES,
        default=DEFAULT_PURPOSE,
        help="训练目标：style 画风（只保留笔触/线条等 general 标签，剔除角色标签）；"
        "character 角色（角色标签放 caption 最前，建议 --threshold 0.25 多留特征细节）",
    )
    tag_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"WD14 模型：内置别名（{', '.join(MODEL_REPOS)}）或完整 HF repo id（默认 {DEFAULT_MODEL}）",
    )
    tag_parser.add_argument(
        "--threshold",
        type=float,
        default=GENERAL_THRESHOLD,
        help=f"general 标签的置信度阈值（默认 {GENERAL_THRESHOLD}）",
    )
    tag_parser.add_argument(
        "--character-threshold",
        type=float,
        default=CHARACTER_THRESHOLD,
        help=f"character 标签的置信度阈值（默认 {CHARACTER_THRESHOLD}）",
    )
    tag_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="推理批大小（默认取 profile 的 tag.batch_size，再默认 8）",
    )
    tag_parser.add_argument(
        "--profile",
        default=None,
        help="运行环境 profile 名（缺省按 .profiles/ 规则自动选择）",
    )
    tag_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="已存在的 .txt 也重新生成（默认跳过，增量打标）",
    )
    tag_parser.set_defaults(func=cmd_tag)

    prepare_parser = sub.add_parser(
        "prepare", help="把图片+caption 转换成 kohya 训练结构和 manifest"
    )
    prepare_parser.add_argument(
        "dataset", help="数据集名（解析为 assets/data/<dataset>）或目录路径"
    )
    prepare_parser.add_argument(
        "--repeats", type=int, default=1, help="kohya 概念目录名中的重复次数（默认 1）"
    )
    prepare_parser.add_argument(
        "--ratings",
        nargs="*",
        default=None,
        metavar="RATING",
        help="只保留指定评分的图片，如 --ratings s g（默认全部保留）",
    )
    prepare_parser.add_argument(
        "--trigger-word",
        default=None,
        help="可选触发词，加在 caption 开头（默认无触发词）",
    )
    prepare_parser.add_argument(
        "--keep-artist-tag",
        action="store_true",
        help="caption 保留 artist 标签（默认去掉，画风隐式训练）",
    )
    prepare_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="切出验证集的比例，如 0.1（默认 0 不切分）；"
        "按 image_id 确定性划分，val 图放 <dataset>/val/ 不参与训练，供 eval 评估泛化",
    )
    prepare_parser.set_defaults(func=cmd_prepare)

    eval_parser = sub.add_parser(
        "eval",
        help="用 CLIP 指标评估各 checkpoint 的拟合程度，推荐最佳 epoch（需 inference 依赖组）",
    )
    eval_parser.add_argument(
        "experiment",
        help="实验名（解析为 assets/experiments/<experiment>）或实验目录路径",
    )
    eval_parser.add_argument(
        "--dataset",
        default=None,
        help="数据集名或目录路径（默认从 config.toml 的 train_data_dir 推断）",
    )
    eval_parser.add_argument(
        "--epochs",
        default=None,
        metavar="SEL",
        help='评估的 epoch 子集，如 "1-6,9"（默认全部）',
    )
    eval_parser.add_argument(
        "--prompts",
        default=None,
        help="提示词文件路径（默认 <实验>/prompts.txt，不存在时用内置默认）",
    )
    eval_parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=1,
        help="每个 prompt 生成几张样例（默认 1；更多张多样性指标更稳）",
    )
    eval_parser.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULT_BASE_SEED,
        help=f"基础随机种子（默认 {DEFAULT_BASE_SEED}）；同一 prompt 跨 checkpoint 固定同 seed",
    )
    eval_parser.add_argument(
        "--clip-model",
        default=None,
        choices=sorted(CLIP_REPOS),
        help=f"打分用 CLIP 模型（默认取 profile 的 eval.clip_model，再默认 {DEFAULT_CLIP}）",
    )
    eval_parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help=f"样例生成边长（默认取 profile 的 eval.resolution，再默认 {DEFAULT_RESOLUTION}；"
             "1024 更贴近训练但更慢）",
    )
    eval_parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="采样步数")
    eval_parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE, help="CFG 引导系数")
    eval_parser.add_argument("--device", default="cuda", help="推理设备（默认 cuda）")
    eval_parser.add_argument(
        "--dtype", default="fp16", choices=["fp16", "bf16", "fp32"], help="推理精度（默认 fp16）"
    )
    eval_parser.add_argument(
        "--offload",
        default=None,
        choices=["sequential", "model", "none"],
        help="显存策略（默认取 profile 的 eval.offload，再默认 sequential 逐层 offload，"
             "6GB 卡实测最快；model=手动分相、UNet 常驻，显存 ≥ 8GB 时用）",
    )
    eval_parser.add_argument(
        "--profile",
        default=None,
        help="运行环境 profile 名（缺省按 .profiles/ 规则自动选择）",
    )
    eval_parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="不生成样例图，只用现有 samples/eval/ 下的图重新打分",
    )
    eval_parser.add_argument(
        "--force",
        action="store_true",
        help="样例图已存在也重新生成（默认跳过，增量可重跑）",
    )
    eval_parser.set_defaults(func=cmd_eval)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
