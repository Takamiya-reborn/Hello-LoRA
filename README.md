<div align="center">

# Hello-LoRA

**画风 / 画师 LoRA 生产线 —— 从图片采集到可用的 `.safetensors` 单文件**

</div>

## 📋 目录

- [技术栈](#-技术栈)
- [流水线](#-流水线)
- [目录结构](#-目录结构)
- [环境搭建](#-环境搭建)
- [详细文档](#-详细文档)

---

## 🧰 技术栈

| 分类     | 技术                                                                                                                                                                                                | 用途                                                   |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| 语言     | ![Python](https://img.shields.io/badge/Python_3.11+-3776AB?logo=python&logoColor=white)                                                                                                             | 主语言                                                 |
| 包管理   | ![uv](https://img.shields.io/badge/uv-DE5FE9?logo=uv&logoColor=white)                                                                                                                               | 依赖管理与虚拟环境，主环境不含 torch                   |
| 训练     | ![PyTorch](https://img.shields.io/badge/PyTorch_cu128-EE4C2C?logo=pytorch&logoColor=white)                                                                                                          | 训练引擎（kohya sd-scripts submodule，独立 venv）      |
| 数据处理 | ![Polars](https://img.shields.io/badge/Polars-0F172A?logo=polars&logoColor=white) ![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?logo=duckdb&logoColor=black)                                 | manifest 统计 / 扫描报表查询                           |
| 图像处理 | ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white) ![Pillow](https://img.shields.io/badge/Pillow-9A4C50) ![imagehash](https://img.shields.io/badge/imagehash-4A90D9) | 解码、校验、感知哈希去重                               |
| 模型推理 | ![ONNX](https://img.shields.io/badge/ONNX_Runtime-005CED?logo=onnx&logoColor=white) ![transformers](https://img.shields.io/badge/transformers-FFD21E?logo=huggingface&logoColor=black)              | WD14 打标（ONNX）；CLIP 评估、样例生成（inference 组） |
| 测试     | ![pytest](https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white)                                                                                                                   | 单元测试                                               |

---

## 🚂 流水线

| 步骤          | 命令 / 工具                                         | 一句话说明                                        | 文档                                     |
| ------------- | --------------------------------------------------- | ------------------------------------------------- | ---------------------------------------- |
| 1. 图片采集   | `scripts/danbooru.py` / `konachan.py` / `kaggle.py` | 从 booru / Kaggle 拉图进 `assets/data/<dataset>/` | [docs/01-collect.md](docs/01-collect.md) |
| 2. 清洗       | `uv run hello-lora scan <dataset>`                  | 扫描图片生成 manifest 与质检报告（增量可重跑）    | [docs/02-scan.md](docs/02-scan.md)       |
| 3. 去重       | 人工确认                                            | 按 sha256 / phash 报告删重复                      | [docs/03-dedupe.md](docs/03-dedupe.md)   |
| 4. 标注       | `uv run hello-lora tag <dataset>`                   | WD14 打标，生成同名 `.txt` caption                | [docs/04-tag.md](docs/04-tag.md)         |
| 5. 数据集验证 | `uv run hello-lora prepare <dataset>`               | 生成 kohya 训练结构，可选切验证集                 | [docs/05-prepare.md](docs/05-prepare.md) |
| 6. 训练       | sd-scripts（独立 venv）                             | 按实验 `config.toml` 训练                         | [docs/06-train.md](docs/06-train.md)     |
| 7. 调参       | 复制 `config.toml` 起新实验                         | 🚧 规划中                                         | [docs/07-tune.md](docs/07-tune.md)       |
| 8. 评估       | `uv run hello-lora eval <experiment>`               | CLIP 指标判过拟合 / 欠拟合，推荐最佳 epoch        | [docs/08-eval.md](docs/08-eval.md)       |
| 9. 导出       | 人工挑选                                            | 最优 checkpoint 拷入 `assets/output/`             | [docs/09-export.md](docs/09-export.md)   |

每个阶段的数据集参数填 `assets/data/` 下的目录名即可（传已存在的目录路径则原样使用）；
涉及随机性的步骤支持 `--seed`。各阶段的产物全部归属数据集 / 实验目录自身，自包含，
增量可重跑。

---

## 📁 目录结构

```text
Hello-LoRA/
├─ assets/
│  ├─ models/
│  │  ├─ ckpt/              # 基础模型，仅 .safetensors
│  │  ├─ clip/              # CLIP 评估打分模型（tokenizer 预置，权重首次运行自动下载）
│  │  └─ onnx/              # WD14 打标等辅助模型
│  ├─ output/               # 最终产物：挑好的 LoRA .safetensors，可整体拷走
│  ├─ data/<dataset>/       # 一个数据集一个目录，自包含（数据 = 输入）
│     ├─ *.jpg / *.txt      # 图片 + 打标产物（同名 .txt caption）
│     ├─ metadata.jsonl     # danbooru/konachan 采集脚本生成（可选）
│     ├─ train/<repeats>_<concept>/
│     ├─ val/               # --val-ratio 切出的验证集（不参与训练，评估用）
│     └─ manifest/          # 各阶段清单：scan / tag / prepare.parquet
│  └─ experiments/<id>/     # 训练实验（数据集平级）：config.toml、日志、checkpoints、
│                           #   manifest/eval.parquet + 评估报告、samples/eval/ 样例
├─ docs/                    # 流水线各步骤详细文档
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

训练引擎 sd-scripts 用独立 venv，分平台安装步骤见 [docs/environment.md](docs/environment.md)。

---

## 📚 详细文档

- [1. 图片采集](docs/01-collect.md) · [2. 清洗](docs/02-scan.md) · [3. 去重](docs/03-dedupe.md) · [4. 标注](docs/04-tag.md)
- [5. 数据集验证](docs/05-prepare.md) · [6. 训练](docs/06-train.md) · [7. 调参](docs/07-tune.md) · [8. 评估](docs/08-eval.md) · [9. 导出](docs/09-export.md)
- [环境搭建（含 sd-scripts 分平台安装）](docs/environment.md)
