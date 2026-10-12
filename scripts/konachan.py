"""从 Konachan 批量下载指定标签的图片到 assets/data/（用法见 README）。

- 独立可执行，无本地模块依赖（仅 requests / tqdm）
- 基于 Moebooru API（与 Danbooru API 兼容），默认走 konachan.net 镜像：
  konachan.com 的 API 目前被 Cloudflare 质询拦截，非浏览器客户端一律 403，
  而 konachan.net 是同一批帖子的 SFW 镜像（不含 explicit），脚本可正常访问
- 自动翻页，目录内图片总数不超过 --limit（已有图片计入，只补差额）
- 已存在的文件按 md5 命名直接跳过，可断点续传
- 下载完成后按 API 给出的 md5 校验内容（--sample 下载的是重压缩版本，
  内容 md5 与帖子 md5 不同，跳过校验），不一致自动重下一次
- 429/5xx/连接错误由 urllib3 Retry 自动退避重试
- 帖子元数据追加写入输出目录的 metadata.jsonl，供后续清洗阶段使用
- 默认只下载静态图片（jpg/png），--all-media 才包含动图等
- --sample 下载 1500px 压缩版而非原图（原图普遍 10MB+，采集画风 LoRA 素材够用；
  注意此时文件名中的 md5 是原图指纹，与文件内容不一致）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

MIRRORS = {
    "com": "https://konachan.com",  # 全量站，API 被 Cloudflare 拦截，暂不可用
    "net": "https://konachan.net",  # SFW 镜像（无 explicit），默认
}
PAGE_SIZE = 100  # Moebooru API 单页常用上限
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SAMPLE_EXT = ".jpg"
# 匿名请求间隔，避免触发限流
API_DELAY = 1.0
DOWNLOAD_DELAY = 0.3
# 单次下载 md5 校验失败后的重试次数
DOWNLOAD_ATTEMPTS = 2

RAW_DIR = Path(__file__).resolve().parent.parent / "assets" / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 Konachan 批量下载指定标签的图片",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tags",
        required=True,
        help='标签查询，多个用逗号分隔（AND 组合），值需加引号如 "<tag1>,<tag2>"',
    )
    parser.add_argument(
        "--rating",
        choices=["safe", "questionable", "explicit"],
        help="按分级过滤（Moebooru 只有 s/q/e 三级，拼成 rating:x 标签）",
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
        "--mirror",
        choices=["com", "net"],
        default="net",
        help="镜像站：net 为 SFW 镜像（默认），com 被 Cloudflare 拦截暂不可用",
    )
    parser.add_argument(
        "--sample", action="store_true", help="下载 1500px sample 版而非原图"
    )
    parser.add_argument(
        "--all-media", action="store_true", help="包含动图等非静态图片文件"
    )
    parser.add_argument(
        "--proxy",
        default="http://127.0.0.1:7892",
        help="HTTP 代理地址（国内直连 konachan 不通），传空字符串禁用",
    )
    return parser.parse_args()


def build_tags(args: argparse.Namespace) -> list[str]:
    # 容错：cmd 不识别单引号，粘贴时可能把引号带进值里
    tags = [
        t.strip().strip("\"'").lower().replace(" ", "_")
        for t in args.tags.split(",")
        if t.strip()
    ]
    if args.rating:
        # Moebooru 的 rating 标签用单字母：s/q/e
        tags.append(f"rating:{args.rating[0]}")
    return tags


def build_session(proxy: str | None = None) -> requests.Session:
    """构建带代理 / 自动重试的会话。

    429/5xx/连接错误交给 urllib3 Retry 退避重试，Retry-After 头（含
    HTTP-date 格式）由其解析并遵守，无需手动处理限流。
    """
    session = requests.Session()
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    session.headers["User-Agent"] = "Hello-LoRA-dataset-collector/0.1"
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_page(
    session: requests.Session, base: str, tags: list[str], page: int
) -> list[dict]:
    """GET 一个 API 页面并返回 JSON；限流 / 瞬时错误由 Retry 适配器处理。"""
    resp = session.get(
        f"{base}/post.json",
        params={"tags": " ".join(tags), "limit": PAGE_SIZE, "page": page},
        timeout=30,
    )
    resp.raise_for_status()
    time.sleep(API_DELAY)
    return resp.json()


def pick_url(post: dict, sample: bool) -> str | None:
    """按 sample 开关返回下载地址；sample 不存在时回退原图。"""
    if sample:
        url = post.get("sample_url")
        if url:
            return url
    return post.get("file_url")


def download_file(
    session: requests.Session, url: str, dest: Path, expected_md5: str | None = None
) -> bool:
    """下载单个文件到 dest，已存在则跳过。

    expected_md5 给定时校验内容一致性（booru 原图有效；压缩版如 --sample
    的内容 md5 与帖子 md5 不同，须传 None 跳过校验）。校验失败删除重下，
    重试后仍失败返回 False，由调用方记为跳过。返回是否实际下载成功。
    """
    if dest.exists():
        return False
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        resp = session.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        digest = hashlib.md5()
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                digest.update(chunk)
        if expected_md5 is None or digest.hexdigest() == expected_md5:
            tmp.rename(dest)
            time.sleep(DOWNLOAD_DELAY)
            return True
        tqdm.write(f"md5 校验失败（第 {attempt}/{DOWNLOAD_ATTEMPTS} 次）: {dest.name}")
        tmp.unlink()
    return False


def count_existing(out_dir: Path, all_media: bool) -> int:
    """统计目录内已下载的图片数（不含元数据文件和 .part 临时文件）。"""
    if all_media:
        return sum(
            1
            for p in out_dir.iterdir()
            if p.is_file() and p.name != "metadata.jsonl" and p.suffix != ".part"
        )
    return sum(1 for p in out_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_seen_md5(meta_path: Path) -> set[str]:
    """读取已写入 metadata.jsonl 的 md5，续传时避免重复写。"""
    seen: set[str] = set()
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(json.loads(line)["md5"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def main() -> None:
    # Windows GBK 控制台下避免中文输出乱码
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    tags = build_tags(args)
    base = MIRRORS[args.mirror]
    out_dir = args.out or RAW_DIR / "_".join(t.split(":")[-1] for t in tags)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"

    existing = count_existing(out_dir, args.all_media)
    target = max(0, args.limit - existing)

    session = build_session(args.proxy or None)

    print(f"镜像站: {base}")
    print(f"标签查询: {' '.join(tags)}")
    print(f"输出目录: {out_dir}")
    print(f"目录已有 {existing} 张，上限 {args.limit} 张，本次待下载 {target} 张")
    if args.mirror == "com":
        print("警告: konachan.com 的 API 目前被 Cloudflare 质询拦截，请求大概率失败。")

    seen_md5 = load_seen_md5(meta_path)

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
                posts = fetch_page(session, base, tags, page)
            except requests.RequestException as e:
                sys.exit(f"请求 Konachan API 失败（第 {page} 页）: {e}")

            if not posts:
                print("没有更多结果了。")
                break

            for post in posts:
                if downloaded >= target:
                    break
                md5 = post.get("md5")
                url = pick_url(post, args.sample)
                if not url or not md5:
                    skipped += 1  # 被删除的帖子没有下载地址
                    continue
                is_sample = url == post.get("sample_url")
                ext = SAMPLE_EXT if is_sample else Path(url).suffix.lower()
                if not args.all_media and ext not in IMAGE_EXTS:
                    skipped += 1
                    continue

                dest = out_dir / f"{md5}{ext}"
                exists = dest.exists()
                try:
                    # sample 是重压缩版本，内容 md5 与帖子 md5 不同，不校验
                    ok = exists or download_file(
                        session,
                        url,
                        dest,
                        expected_md5=None if is_sample else md5,
                    )
                except requests.RequestException as e:
                    tqdm.write(f"下载失败 {md5}: {e}")
                    continue
                if not ok:
                    skipped += 1  # md5 校验重试后仍不匹配
                    continue
                if not exists:
                    downloaded += 1
                    bar.update(1)
                # 已存在的文件若 metadata 缺失（如续传时清单丢失）也补录
                if md5 not in seen_md5:
                    meta_file.write(json.dumps(post, ensure_ascii=False) + "\n")
                    seen_md5.add(md5)

            page += 1
            if page > 1000:
                print("已达第 1000 页上限。")
                break

    print(f"完成: 新下载 {downloaded} 张，跳过 {skipped} 个，输出目录: {out_dir}")


if __name__ == "__main__":
    main()
