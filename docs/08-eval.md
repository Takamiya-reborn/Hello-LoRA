# 8. 评估（拟合评估：过拟合 / 欠拟合）

对每个 epoch checkpoint 用固定 prompt 事后生成样例图，再算 CLIP 指标判断拟合程度，
输出逐 epoch 指标表和推荐 epoch。数据集名、底模路径都从实验的 config.toml 自动读取。

```bash
# 首次使用需先加装推理依赖（torch cu128 + diffusers + transformers + peft）
uv sync --group inference

uv run hello-lora eval <experiment>
#    可选: --epochs 1-6,9 只评估部分 epoch, --dataset <dataset> 指定数据集
#          （默认从 config.toml 的 train_data_dir 推断）, --prompts <file> 换提示词,
#          --samples-per-prompt 3, --clip-model vit-l|vit-bigg,
#          --resolution 768, --steps 30, --offload model|sequential|none,
#          --skip-generation 只重打分不出图, --force 已有样例也重新生成
```

## 指标口径

base = 底模无 LoRA 的基线，e = 某 epoch：

| 指标 | 含义 | 拟合信号 |
| --- | --- | --- |
| `style_sim_train` | 样例与训练图 CLIP 图像相似度的均值（画风保真） | 低于 base + 0.02 → **欠拟合**（LoRA 没起效） |
| `style_sim_val` / gap | 与验证集图的相似度；gap = train − val | gap 随 epoch 增大 → **过拟合**（泛化变差） |
| `clip_t` | 图文相似度（prompt 遵循度） | 后期下跌 → 语义可控性被过拟合吃掉 |
| `diversity` | 同 checkpoint 样例两两相似度均值 | 相对 base 跌幅 > 0.05 → 输出塌缩 |
| `nearest_train_sim` | 与最近一张训练图的相似度 | > 0.90 → 复刻训练图（记忆化） |

判定为启发式规则，报告里逐 epoch 列出全部指标供人工复核——曲线上
"style 上升趋平、clip_t 开始回落"的拐点通常就是甜点。

## 评估提示词

默认读 `assets/experiments/<id>/prompts.txt`（一行一条
`<id>|style或unseen|<prompt>`，`#` 注释），没有该文件时用内置默认（不含触发词；
用触发词训练的实验请自定义）。

- `style` 条目：训练集里出现过的主体域，衡量画风习得；
- `unseen` 条目：训练集没有的主体/场景，衡量泛化——过拟合时这类 prompt 的指标率先崩。

同一 prompt 在所有 checkpoint 上用同一个确定性 seed（由 prompt_id 派生），
保证跨 epoch 曲线可比。

## 产物

全部写入实验目录：

```text
assets/experiments/<id>/
├─ manifest/eval.parquet        # 每 checkpoint × 每 prompt × 每 seed 一行
├─ manifest/eval.report.json    # 逐 epoch 汇总指标 + 判定 + 推荐 epoch
├─ manifest/clip_ref_cache.npz  # 参考图 embedding 缓存（增量重跑不重算）
└─ samples/eval/<checkpoint>/<prompt_id>_seed<seed>.png
```

## 运行成本与增量

- 中断重跑自动跳过已有样例图；`--skip-generation` 可只对现有样例重打分（秒级）；
- 底模只加载一次，checkpoint 间只换 LoRA 权重；默认 `--offload sequential`
  （逐层搬运，显存仅 ~1.9GB，不会触发驱动的 sysmem 溢出，RTX 4050 6GB
  实测 768px 约 1.6s/步、每张样例约 1 分钟）；显存 ≥ 8GB 可换
  `--offload model`（UNet 常驻）或 `none`；
- ⚠️ fp16 UNet 常驻约 5.0GB，6GB 卡上放不下会**静默溢出到系统内存**、
  速度掉一个数量级（GPU 占用 100% 但功耗只有 30W 左右）——慢就换 sequential；
- CLIP 打分模型缓存到 `assets/models/clip/<org>_<repo>/`（tokenizer 已预置，
  权重首次运行自动下载，`HF_ENDPOINT=https://hf-mirror.com` 走镜像）；
- **无验证集时自动降级**：只用记忆化 / 多样性 / clip_t 三个信号判断过拟合，
  报告会提示下次 prepare 加 `--val-ratio 0.1`。
