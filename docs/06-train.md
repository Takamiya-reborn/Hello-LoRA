# 6. 训练（kohya sd-scripts）

## 实验配置

每个实验一个 `assets/experiments/<id>/config.toml`，手写即可（字段要点见下文）。
熟悉 sd-scripts 的话，也可以用它自带的 `--output_config` 从命令行参数生成
config 快照（写完即退出，再改再训）：

```bash
tools/sd-scripts/.venv/Scripts/python.exe tools/sd-scripts/train_network.py \
    <参数...> --output_config "assets/experiments/<id>/config.toml"
```

## 运行训练

```bash
tools/sd-scripts/.venv/Scripts/python.exe tools/sd-scripts/train_network.py \
    --config_file "assets/experiments/<id>/config.toml"
```

Linux/macOS 把 `Scripts` 换成 `bin`。

> ⚠️ config.toml 中的路径为仓库根相对路径（kohya 按当前工作目录解析），**必须从仓库根目录运行**。

## 实验布局

每个实验一个目录 `assets/experiments/<id>/`，自包含 config、日志与产物：

```text
assets/experiments/<id>/
├─ config.toml       # 训练配置（底模、数据、网络、优化器、输出）
├─ train.log         # 控制台输出
├─ log/              # tensorboard 事件（uv run tensorboard --logdir log 查看）
├─ output/           # 按 epoch 保存的 checkpoint（save_every_n_epochs）
└─ samples/          # 评估样例图（由第 8 步生成）
```

## config.toml 要点（6GB 显存预设）

- `fp8_base = true` + `gradient_checkpointing` + `cache_latents_to_disk` +
  `cache_text_encoder_outputs_to_disk`：低显存训练的核心开关；
- `cache_text_encoder_outputs_to_disk` 与 `shuffle_caption` 互斥（文本编码器输出
  已缓存，caption 顺序随机化无效），二选一时优先保 TE 缓存；
- `save_every_n_epochs = 1`：逐 epoch 留 checkpoint，供第 8 步评估挑最佳。

环境搭建见 [00-environment.md](00-environment.md)。
