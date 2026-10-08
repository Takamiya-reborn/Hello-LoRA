"""从 Danbooru 批量下载指定标签或艺术家的图片到 assets/data/（用法见 README）。

- 自动翻页，目录内图片总数不超过 --limit（已有图片计入，只补差额）
- 已存在的文件按 md5 命名直接跳过，可断点续传
- 帖子元数据追加写入输出目录的 metadata.jsonl，供后续清洗阶段使用
- 默认只下载静态图片（jpg/png/webp），--all-media 才包含动图和视频
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

API_BASE = "https://danbooru.donmai.us"
PAGE_SIZE = 200  # API 单页上限
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
# 匿名请求间隔，避免触发 429
API_DELAY = 1.0
DOWNLOAD_DELAY = 0.3

RAW_DIR = Path(__file__).resolve().parent.parent / "assets" / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 Danbooru 批量下载指定标签或艺术家的图片",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument(
        "--tags",
        help='标签查询，多个用逗号分隔（AND 组合），值需加引号如 "<tag1>,<tag2>"',
    )
    query.add_argument(
        "--artist",
        help='艺术家标签名，值需加引号如 "<artist>_(suffix)"（括号在 shell 中有特殊含义）',
    )
    parser.add_argument(
        "--rating",
        choices=["general", "sensitive", "questionable", "explicit"],
        help="按分级过滤（拼成 rating:xxx 标签）",
    )
    parser.add_argument(
        "--limit", type=int, default=100, help="目录内图片总数上限（已有图片计入）"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出目录（默认 assets/data/<标签组合>）",
    )
    parser.add_argument(
        "--username", help="Danbooru 用户名（下超过 2 个标签的组合时需要）"
    )
    parser.add_argument("--api-key", help="Danbooru API Key（在账号设置页面生成）")
    parser.add_argument(
        "--all-media", action="store_true", help="包含动图/视频等非静态图片文件"
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help="HTTP 代理地址，例如 http://127.0.0.1:<port>（国内直连 Danbooru 超时时使用）",
    )
    return parser.parse_args()


def build_tags(args: argparse.Namespace) -> list[str]:
    tags = args.tags.split(",") if args.tags else [args.artist]
    # 容错：cmd 不识别单引号，粘贴时可能把引号带进值里
    tags = [t.strip().strip("\"'").lower().replace(" ", "_") for t in tags if t.strip()]
    if args.rating:
        tags.append(f"rating:{args.rating}")
    return tags


def fetch_page(session: requests.Session, tags: list[str], page: int) -> list[dict]:
    resp = session.get(
        f"{API_BASE}/posts.json",
        params={"tags": " ".join(tags), "limit": PAGE_SIZE, "page": page},
        timeout=30,
    )
    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", "30"))
        tqdm.write(f"触发限流，等待 {retry_after} 秒后重试 ...")
        time.sleep(retry_after)
        return fetch_page(session, tags, page)
    resp.raise_for_status()
    time.sleep(API_DELAY)
    return resp.json()


def download_file(session: requests.Session, post: dict, dest: Path) -> bool:
    """下载单个文件，已存在则跳过。返回是否实际下载。"""
    if dest.exists():
        return False
    resp = session.get(post["file_url"], timeout=120, stream=True)
    resp.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    tmp.rename(dest)
    time.sleep(DOWNLOAD_DELAY)
    return True


def count_existing(out_dir: Path, all_media: bool) -> int:
    """统计输出目录中已下载的图片数（不含元数据文件和 .part 临时文件）。"""
    if all_media:
        return sum(
            1
            for p in out_dir.iterdir()
            if p.is_file() and p.name != "metadata.jsonl" and p.suffix != ".part"
        )
    return sum(1 for p in out_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def main() -> None:
    # Windows GBK 控制台下避免中文输出乱码
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    tags = build_tags(args)
    out_dir = args.out or RAW_DIR / "_".join(t.split(":")[-1] for t in tags)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"

    existing = count_existing(out_dir, args.all_media)
    target = max(0, args.limit - existing)

    auth = (args.username, args.api_key) if args.username and args.api_key else None
    session = requests.Session()
    session.auth = auth
    if args.proxy:
        session.proxies = {"http": args.proxy, "https": args.proxy}
    session.headers["User-Agent"] = "Hello-LoRA-dataset-collector/0.1"

    print(f"标签查询: {' '.join(tags)}")
    print(f"输出目录: {out_dir}")
    print(f"目录已有 {existing} 张，上限 {args.limit} 张，本次待下载 {target} 张")
    if auth is None and len(tags) > 2:
        print("警告: 匿名请求最多支持 2 个标签组合，多余标签可能被服务端拒绝。")

    # 续传时避免向 metadata.jsonl 重复写入
    seen_md5: set[str] = set()
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            try:
                seen_md5.add(json.loads(line)["md5"])
            except (json.JSONDecodeError, KeyError):
                continue

    downloaded, skipped = 0, 0
    page = 1
    if target == 0:
        print("已达上限，无需下载。")
        return
    with (
        meta_path.open("a", encoding="utf-8") as meta_file,
        tqdm(total=target, desc="下载中", unit="张") as bar,
    ):
        while downloaded < target:
            try:
                posts = fetch_page(session, tags, page)
            except requests.RequestException as e:
                sys.exit(f"请求 Danbooru API 失败（第 {page} 页）: {e}")

            if not posts:
                print("没有更多结果了。")
                break

            for post in posts:
                if downloaded >= target:
                    break
                file_url = post.get("file_url")
                md5 = post.get("md5")
                if not file_url or not md5:
                    skipped += 1  # 付费墙/被删除的帖子没有 file_url
                    continue
                ext = Path(file_url).suffix.lower()
                if not args.all_media and ext not in IMAGE_EXTS:
                    skipped += 1
                    continue

                dest = out_dir / f"{md5}{ext}"
                try:
                    if download_file(session, post, dest):
                        downloaded += 1
                    else:
                        skipped += 1
                except requests.RequestException as e:
                    tqdm.write(f"下载失败 {md5}: {e}")
                    continue
                bar.update(1)
                if md5 not in seen_md5:
                    meta_file.write(json.dumps(post, ensure_ascii=False) + "\n")
                    seen_md5.add(md5)

            page += 1
            if page > 1000:
                print("已达匿名/免费账号的第 1000 页上限。")
                break

    print(f"完成: 新下载 {downloaded} 张，跳过 {skipped} 个，输出目录: {out_dir}")


if __name__ == "__main__":
    main()
