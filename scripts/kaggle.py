"""从 Kaggle 下载数据集并复制到 assets/data/。

用法:
    uv run python scripts/kaggle.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

DATASET = "bisheshkhanalcs26/genshin-impact"
RAW_DIR = Path(__file__).resolve().parent.parent / "assets" / "data"


def main() -> None:
    print(f"正在下载数据集: {DATASET} ...")
    cache_path = Path(kagglehub.dataset_download(DATASET))
    print(f"kagglehub 缓存路径: {cache_path}")

    dest = RAW_DIR / DATASET.split("/")[-1]
    if dest.exists():
        print(f"目标目录已存在，跳过复制: {dest}")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(cache_path, dest)
        print(f"已复制到: {dest}")

    files = sorted(p for p in dest.rglob("*") if p.is_file())
    print(f"共 {len(files)} 个文件:")
    for p in files[:20]:
        print(f"  {p.relative_to(dest)} ({p.stat().st_size / 1024:.1f} KB)")
    if len(files) > 20:
        print(f"  ... 以及另外 {len(files) - 20} 个文件")


if __name__ == "__main__":
    main()
