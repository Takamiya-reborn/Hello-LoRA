"""profiles 模块测试：模板生成、选择规则、默认值回退与 TOML 平铺写入。"""

import os
from pathlib import Path

import pytest

from hello_lora.profiles import (
    TEMPLATE,
    ensure_default,
    get as profile_get,
    list_profiles,
    load_profile,
    resolve_profile,
)


def write_profile(directory: Path, name: str, content: str) -> Path:
    path = directory / f"{name}.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_ensure_default_creates_template(tmp_path: Path) -> None:
    ensure_default(tmp_path)
    assert list_profiles(tmp_path) == ["local"]
    profile = load_profile("local", tmp_path)
    # 模板只覆盖 hello-lora 自己的默认值（代理/镜像/打标/评估），不含训练参数
    assert profile["eval"]["offload"] == "sequential"
    assert profile["tag"]["batch_size"] == 8
    assert "train" not in profile


def test_ensure_default_idempotent(tmp_path: Path) -> None:
    """目录里已有 profile 时不覆盖（用户改过的配置不能被模板冲掉）。"""
    custom = write_profile(tmp_path, "mine", 'description = "custom"\n')
    ensure_default(tmp_path)
    assert custom.read_text(encoding="utf-8") == 'description = "custom"\n'
    assert list_profiles(tmp_path) == ["mine"]


def test_resolve_single_profile_no_prompt(tmp_path: Path) -> None:
    write_profile(tmp_path, "cloud", '[eval]\noffload = "none"\n')
    name, profile = resolve_profile(directory=tmp_path, interactive=False)
    assert name == "cloud"
    assert profile["eval"]["offload"] == "none"


def test_resolve_explicit_missing_lists_available(tmp_path: Path) -> None:
    write_profile(tmp_path, "local", "")
    with pytest.raises(SystemExit, match="local"):
        resolve_profile("cloud", directory=tmp_path)


def test_resolve_multiple_without_flag_needs_choice(tmp_path: Path) -> None:
    write_profile(tmp_path, "a", "")
    write_profile(tmp_path, "b", "")
    # 非交互环境不猜：报错并列出可用项
    with pytest.raises(SystemExit, match="a, b"):
        resolve_profile(directory=tmp_path, interactive=False)
    # 交互环境按输入序号选
    import builtins

    original = builtins.input
    builtins.input = lambda *_: "2"
    try:
        name, _ = resolve_profile(directory=tmp_path, interactive=True)
    finally:
        builtins.input = original
    assert name == "b"


def test_get_dotted_fallback(tmp_path: Path) -> None:
    ensure_default(tmp_path)
    profile = load_profile("local", tmp_path)
    assert profile_get(profile, "tag.batch_size", 8) == 8
    assert profile_get(profile, "eval.clip_model", "vit-l") == "vit-l"
    assert profile_get(profile, "nope.missing", "fallback") == "fallback"
    # 模板里 network.proxy 为空串 -> 取 fallback（直连）
    assert profile_get(profile, "network.proxy", None) is None


def test_apply_network_env_respects_explicit_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hello_lora.profiles import apply_network_env

    profile = {"network": {"hf_endpoint": "https://hf-mirror.com"}}
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    apply_network_env(profile)
    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
    # 用户显式设置的优先，不被 profile 覆盖
    monkeypatch.setenv("HF_ENDPOINT", "https://explicit.example.com")
    apply_network_env(profile)
    assert os.environ["HF_ENDPOINT"] == "https://explicit.example.com"


def test_template_is_valid_toml() -> None:
    import tomllib

    data = tomllib.loads(TEMPLATE)
    assert "network" in data and "eval" in data and "tag" in data
