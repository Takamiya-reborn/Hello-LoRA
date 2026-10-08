<div align="center">

# Hello-LoRA

**画风 / 画师 LoRA 生产线 —— 从图片采集到可用的 `.safetensors` 单文件**

</div>

## 📋 目录

- [技术栈](#-技术栈)
- [流水线](#-流水线)
- [目录结构](#-目录结构)
- [环境搭建](#-环境搭建)

---

## 🧰 技术栈

| 分类     | 技术                                                                                                                                                                                                | 用途                                              |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| 语言     | ![Python](https://img.shields.io/badge/Python_3.11+-3776AB?logo=python&logoColor=white)                                                                                                             | 主语言                                            |
| 包管理   | ![uv](https://img.shields.io/badge/uv-DE5FE9?logo=uv&logoColor=white)                                                                                                                               | 依赖管理与虚拟环境，主环境不含 torch              |
| 训练     | ![PyTorch](https://img.shields.io/badge/PyTorch_cu128-EE4C2C?logo=pytorch&logoColor=white)                                                                                                          | 训练引擎（kohya sd-scripts submodule，独立 venv） |
| 数据处理 | ![Polars](https://img.shields.io/badge/Polars-0F172A?logo=polars&logoColor=white) ![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?logo=duckdb&logoColor=black)                                 | manifest 统计 / 扫描报表查询                      |
| 图像处理 | ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white) ![Pillow](https://img.shields.io/badge/Pillow-9A4C50) ![imagehash](https://img.shields.io/badge/imagehash-4A90D9) | 解码、校验、感知哈希去重                          |
| 模型推理 | ![ONNX](https://img.shields.io/badge/ONNX_Runtime-005CED?logo=onnx&logoColor=white)                                                                                                                 | WD14 打标、CLIP 评估                              |
| 测试     | ![pytest](https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white)                                                                                                                   | 单元测试                                          |

---

## 🚂 流水线

<details>
<summary><b>1. 图片采集</b></summary>

```bash
# Danbooru 按艺术家采集（按标签用 --tags "<tag1>,<tag2>"）
uv run python scripts/danbooru.py --artist "<artist>" --limit 200

# Kaggle 数据集采集
uv run python scripts/kaggle.py
```

</details>

<details>
<summary><b>2. 清洗</b></summary>

```bash
uv run hello-lora scan --input "assets/data/<dataset>" --output "assets/data/<dataset>/manifest/scan"
```

生成 manifest 清单 + 扫描报告。

</details>

<details>
<summary><b>3. 去重</b></summary>

由上一步 `scan` 的报告覆盖：完全重复组见报告的 `duplicate_exact_groups`，待人工确认后删除。

</details>

<details>
<summary><b>4. 标注</b></summary>

> 🚧 规划中：计划用 WD14 打标（`assets/models/onnx/`），暂无 CLI。

</details>

<details>
<summary><b>5. 数据集验证</b></summary>

```bash
uv run hello-lora prepare "assets/data/<dataset>"
#    可选: --ratings s g 按分级过滤, --trigger-word <word> 加触发词,
#          --repeats 2 重复次数, --keep-artist-tag 保留画师标签
```

产出 kohya 训练结构 + manifest，损坏/无元数据文件会被跳过并列出原因。

</details>

<details>
<summary><b>6. 训练</b></summary>

```bash
tools/sd-scripts/.venv/Scripts/python.exe tools/sd-scripts/train_network.py \
    --config_file "assets/data/<dataset>/experiments/<id>/config.toml"
```

Linux/macOS 把 `Scripts` 换成 `bin`。

> ⚠️ config.toml 中的路径为仓库根相对路径（kohya 按当前工作目录解析），**必须从仓库根目录运行**。

</details>

<details>
<summary><b>7. 调参</b></summary>

> 🚧 规划中：复制 config.toml 起新实验，对比各 checkpoint。

</details>

<details>
<summary><b>8. 评估</b></summary>

> 🚧 规划中：计划用 CLIP（`assets/models/clip/`），暂无 CLI。

</details>

<details>
<summary><b>9. 导出</b></summary>

人工挑选效果最好的 checkpoint，拷入 `assets/output/`。最终产物：单个 `.safetensors` LoRA 文件。

</details>

---

## 📁 目录结构

```text
Hello-LoRA/
├─ assets/
│  ├─ models/
│  │  ├─ ckpt/              # 基础模型，仅 .safetensors
│  │  ├─ clip/              # CLIP 模型（评估用）
│  │  └─ onnx/              # WD14 打标等辅助模型
│  ├─ output/               # 最终产物：挑好的 LoRA .safetensors，可整体拷走
│  └─ data/<dataset>/       # 一个数据集一个目录，自包含
│     ├─ *.jpg / metadata.jsonl
│     ├─ train/<repeats>_<concept>/
│     ├─ manifest/
│     └─ experiments/<id>/  # config.toml、日志、checkpoints、样例
├─ scripts/                 # 可执行脚本
├─ src/hello_lora/          # Python 包
├─ tests/
└─ tools/sd-scripts/        # 训练引擎（git submodule）
```

---

## ⚙️ 环境搭建

主环境：

```bash
uv sync
```

默认组不含 torch。需要推理 / 评估时加装 `inference` 组（`uv sync --group inference`），
torch 会按平台自动选源：Windows 走 cu128（PyPI torch 是 CPU 版），Linux / macOS 用 PyPI 默认源（自带 CUDA / MPS）。

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
