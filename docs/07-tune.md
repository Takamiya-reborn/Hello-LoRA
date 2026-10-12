# 7. 调参

> 🚧 规划中：复制 config.toml 起新实验（`assets/experiments/<新id>/`），
> 对比各 checkpoint。评估对比能力由第 8 步提供（多实验的 eval.report.json 可横向比较）。

建议每次实验只动一组变量（学习率 / network_dim / 数据量 / 触发词策略），
否则指标差异无法归因。
