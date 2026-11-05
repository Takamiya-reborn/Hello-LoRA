"""运行环境 profile：.profiles/<名字>.toml，一台机器一份。

profile 提供各命令的默认值，CLI flag 永远覆盖。目录整体拷走即迁移，
已被 .gitignore 排除（每台机器内容不同）。

resolve_profile() 的选择规则：
1. 显式指定（--profile <名字>）；
2. 目录里只有一个 profile -> 直接用它；
3. 有多个且未指定 -> 交互式选择（非交互环境报错并列出可用项）；
4. .profiles/ 不存在或为空 -> 先从内置模板生成 local.toml，再按上述规则走。
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

PROFILES_DIR = Path(".profiles")
DEFAULT_PROFILE = "local"

TEMPLATE = """\
# Hello-LoRA 运行环境 profile（文件名即 profile 名，目录整体拷走即迁移）
description = "本地默认模板（按实际环境修改，可复制多份）"

[network]
# 采集脚本代理，空 = 直连；本地走 Clash 等填 http://127.0.0.1:7892
proxy = ""
# HuggingFace 镜像，国内机器填 https://hf-mirror.com（显式设置的环境变量优先于这里）
hf_endpoint = ""

[tag]
# WD14 打标推理批大小
batch_size = 8

[eval]
# 样例生成显存策略：sequential = 逐层 offload（6GB 卡实测最快）；显存 >= 8GB 可用 "none"
offload = "sequential"
resolution = 768
clip_model = "vit-l"
"""


def profiles_dir(directory: Path | None = None) -> Path:
    return directory if directory is not None else PROFILES_DIR


def ensure_default(directory: Path | None = None) -> Path:
    """.profiles/ 不存在或没有任何 profile 时，从内置模板生成 local.toml。"""
    d = profiles_dir(directory)
    d.mkdir(parents=True, exist_ok=True)
    default = d / f"{DEFAULT_PROFILE}.toml"
    if not default.exists() and not list_profiles(directory=d):
        default.write_text(TEMPLATE, encoding="utf-8")
        print(f"已生成默认 profile: {default}（按实际硬件修改后可复制多份）")
    return default


def list_profiles(directory: Path | None = None) -> list[str]:
    return sorted(p.stem for p in profiles_dir(directory).glob("*.toml"))


def load_profile(name: str, directory: Path | None = None) -> dict:
    path = profiles_dir(directory) / f"{name}.toml"
    if not path.is_file():
        available = ", ".join(list_profiles(directory)) or "（无）"
        raise SystemExit(f"profile 不存在: {path}（可用: {available}）")
    with path.open("rb") as f:
        return tomllib.load(f)


def describe(profile: dict) -> str:
    desc = profile.get("description")
    return f" — {desc}" if isinstance(desc, str) and desc else ""


def apply_network_env(profile: dict) -> None:
    """把 [network] 落到进程环境：HF_ENDPOINT（用户显式设置的环境变量优先）。

    需在 huggingface_hub 首次 import 前调用才生效，因此 tag 的 HF 导入是
    函数内懒加载（见 tag.ensure_model）。
    """
    hf = (profile.get("network") or {}).get("hf_endpoint")
    if hf:
        os.environ.setdefault("HF_ENDPOINT", hf)


def resolve_profile(
    name: str | None = None,
    directory: Path | None = None,
    interactive: bool | None = None,
) -> tuple[str, dict]:
    """按模块 docstring 的规则选定 profile，返回 (名字, 内容 dict)。"""
    ensure_default(directory)
    names = list_profiles(directory)
    if name is not None:
        profile = load_profile(name, directory)
    elif len(names) == 1:
        name, profile = names[0], load_profile(names[0], directory)
    else:
        if interactive is None:
            interactive = sys.stdin.isatty()
        if not interactive:
            raise SystemExit(
                f"存在多个 profile 且未指定，请用 --profile 选择（可用: {', '.join(names)}）"
            )
        print("存在多个 profile，请选择:")
        for i, n in enumerate(names, 1):
            print(f"  {i}. {n}{describe(load_profile(n, directory))}")
        raw = input(f"输入序号 [1-{len(names)}，回车 = 1]: ").strip()
        idx = int(raw) if raw.isdigit() and 1 <= int(raw) <= len(names) else 1
        name, profile = names[idx - 1], load_profile(names[idx - 1], directory)
    apply_network_env(profile)
    print(f"profile: {name}{describe(profile)}")
    return name, profile


def get(profile: dict, dotted: str, fallback):
    """按点分路径取 profile 值（'tag.batch_size'）。

    缺失、None 或空串均回退 fallback（模板用空串表示"未设置"，如 proxy = "" 直连）。
    """
    node = profile
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return fallback
        node = node[part]
    return fallback if node is None or node == "" else node
