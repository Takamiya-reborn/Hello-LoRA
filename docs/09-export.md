# 9. 导出

人工挑选效果最好的 checkpoint（参考第 8 步评估报告的推荐 epoch 与 `samples/eval/`
样例图），拷入 `assets/output/`。最终产物：单个 `.safetensors` LoRA 文件，可整体拷走。

建议文件名带上实验 id 与 epoch（如 `nekoda_style_v1-ep6.safetensors`），便于追溯来源。
