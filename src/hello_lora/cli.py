"""Hello-LoRA 命令行入口。

每个数据处理步骤对应一个子命令，统一支持 --input / --output；
涉及随机性的步骤额外支持 --seed。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hello_lora.prepare import prepare_dataset, summarize_dataset
from hello_lora.scan import scan_directory, summarize


def cmd_prepare(args: argparse.Namespace) -> None:
    manifest, train_dir, skipped = prepare_dataset(
        Path(args.dataset_dir),
        repeats=args.repeats,
        ratings=args.ratings,
        trigger_word=args.trigger_word,
        keep_artist_tag=args.keep_artist_tag,
    )
    if skipped:
        print(f"跳过 {len(skipped)} 个文件:")
        for name, reason in skipped[:10]:
            print(f"  {name}: {reason}")
        if len(skipped) > 10:
            print(f"  ... 以及另外 {len(skipped) - 10} 个文件")
    print(f"manifest 已写入: {Path(args.dataset_dir) / 'manifest' / 'image_manifest.parquet'} ({manifest.height} 行)")
    n_images = len(list(train_dir.glob("*.*")))
    print(f"训练目录: {train_dir} -> {n_images - manifest.height} 张图")
    print(summarize_dataset(manifest))


def cmd_scan(args: argparse.Namespace) -> None:
    input_dir = Path(args.input)
    output = Path(args.output)
    if not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在: {input_dir}")

    manifest = scan_directory(input_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(output.with_suffix(".parquet"))
    report = summarize(manifest)
    report_path = output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"清单已写入: {output.with_suffix('.parquet')} ({manifest.height} 个文件)")
    print(f"扫描报告已写入: {report_path}")
    print(
        f"待人工确认: {len(report['needs_review'])} 个；完全重复组: {report['duplicate_exact_groups']}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hello-lora", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    scan_parser = sub.add_parser(
        "scan", help="扫描图片目录，生成 manifest 清单和扫描报告"
    )
    scan_parser.add_argument("--input", required=True, help="待扫描的图片目录")
    scan_parser.add_argument(
        "--output",
        required=True,
        help="输出文件路径（不含扩展名，生成 .parquet 和 .report.json）",
    )
    scan_parser.set_defaults(func=cmd_scan)

    prepare_parser = sub.add_parser(
        "prepare", help="把图片+metadata.jsonl 转换成 kohya 训练结构和 manifest"
    )
    prepare_parser.add_argument("dataset_dir", type=Path, help="数据集目录")
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
    prepare_parser.set_defaults(func=cmd_prepare)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
