# 4. 标注（WD14 打标）

```bash
uv run hello-lora tag <dataset>
#    可选: --purpose style|character 区分训练目标（默认 style）,
#          --model <alias|hf-repo> 换模型, --threshold 0.35, --character-threshold 0.85,
#          --batch-size 8, --overwrite 强制重标
```

用 WD14 模型（ONNX）给每张图打标，生成图片同名 `.txt`（kohya caption 约定），
并写 `<dataset>/manifest/tag.parquet`（含 rating 预测，回填 scan 清单的 `caption_status`）。
标签按置信度降序、下划线转空格；已存在的 `.txt` 默认跳过，可增量重跑。

## 训练目标（--purpose）

- `style` 画风/笔触：只保留 general 标签（笔触、线条、上色方式、构图都在这里），
  剔除 character 标签，避免把角色身份绑进画风；风格本身交给 prepare 的 `--trigger-word` 绑定；
- `character` 角色：character 标签（角色名/服装变体）放 caption 最前，
  建议配合 `--threshold 0.25` 多保留特征细节。

换目标重标需加 `--overwrite`。

## 模型

来自 Hugging Face 的 SmilingWolf 系列，自动下载缓存到 `assets/models/onnx/<repo>/`，
本地齐全后不再联网。内置别名：`wd-vit-v3`（默认）/ `wd-swinv2-v3` / `wd-eva02-large-v3` /
`wd-convnextv2-v2` / `wd-swinv2-v2` / `wd-moat-v2`，也可直接传完整 repo id。

**模型部署要求**：每个模型一个目录，路径把 repo id 里的 `/` 换成 `_`，
**文件名必须是 `model.onnx` 和 `selected_tags.csv`**（缺哪个补哪个，程序按这两个名字查找）：

```text
assets/models/onnx/
├─ SmilingWolf_wd-vit-tagger-v3/
│  ├─ model.onnx            # 模型权重
│  └─ selected_tags.csv     # 标签表（tag_id,name,category,count），与模型配套
└─ SmilingWolf_wd-swinv2-tagger-v3/
   └─ ...
```

- 自动下载走 huggingface_hub，国内网络可设 `HF_ENDPOINT=https://hf-mirror.com` 走镜像；
  手动下载（镜像页/网盘）后按上面的目录和文件名放好即可，程序检测到文件齐全就不再联网；
- 输入尺寸从模型自动读取（`wd-vit-v3` / `wd-swinv2-v3` 为 448，`wd-eva02-large-v3` 为 512），
  不要混用不同模型的标签表；
- 预处理与 sd-scripts 的 tag_images_by_wd14_tagger.py 同口径：
  透明通道合成到白底 → 白边补正方形 → 缩放到模型输入尺寸 → RGB 转 BGR。

## 推理后端

主环境装的是 **CPU 版 onnxruntime**（约 1.5 张/秒，百张图分钟级）。
要 GPU 加速需换装 `onnxruntime-gpu`——Windows 下要求本机装有 CUDA 12.x + cuDNN 9 并在
PATH 中（训练 venv 里的 torch cu128 是独立环境，不会复用其 CUDA 运行库）：

```bash
uv pip install --reinstall onnxruntime-gpu
```

装好后命令会打印实际使用的后端（CUDAExecutionProvider / CPUExecutionProvider）。
