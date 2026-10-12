# 环境搭建

主环境（uv 管理，默认组不含 torch）：

```bash
uv sync
```

需要推理 / 评估（`hello-lora eval`、样例生成）时加装 `inference` 组：

```bash
uv sync --group inference
```

torch 会按平台自动选源：Windows 走 cu128（PyPI torch 是 CPU 版），
Linux / macOS 用 PyPI 默认源（自带 CUDA / MPS）。

## 训练引擎（kohya sd-scripts）

sd-scripts 用独立 venv，避免与主环境 torch 版本冲突：

```bash
git submodule update --init --depth 1

uv venv tools/sd-scripts/.venv --python 3.11
```

按平台安装 torch 和依赖（Windows 路径用 `Scripts`，Linux/macOS 换成 `bin`）：

```bash
# Windows + NVIDIA 显卡：PyPI torch 是 CPU 版，必须走官方 index，且先于 requirements 安装
uv pip install --python tools/sd-scripts/.venv/Scripts/python.exe \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python tools/sd-scripts/.venv/Scripts/python.exe \
    -r tools/sd-scripts/requirements.txt
# 修复 opencv-python 4.10 轮子在 Windows 上的 import 崩溃
uv pip install --python tools/sd-scripts/.venv/Scripts/python.exe \
    --reinstall "opencv-python>=4.11"

# Linux + NVIDIA 显卡：PyPI torch 自带 CUDA，直接装依赖即可
# （需要指定 CUDA 版本时可参照 Windows 加装官方 index 的 torch）
uv pip install --python tools/sd-scripts/.venv/bin/python \
    -r tools/sd-scripts/requirements.txt

# macOS (Apple Silicon)：PyPI torch 自带 MPS，直接装依赖即可
uv pip install --python tools/sd-scripts/.venv/bin/python \
    -r tools/sd-scripts/requirements.txt
```
