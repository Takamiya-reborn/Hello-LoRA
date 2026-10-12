# 5. 数据集验证（prepare）

```bash
uv run hello-lora prepare <dataset>
#    可选: --ratings s g 按分级过滤, --trigger-word <word> 加触发词,
#          --repeats 2 重复次数, --keep-artist-tag 保留画师标签,
#          --val-ratio 0.1 切出 10% 验证集（供评估泛化）
```

产出 kohya 训练结构 + manifest，损坏/无 caption 来源的文件会被跳过并列出原因。

## caption 来源

优先级：打标产物 `<stem>.txt` > `metadata.jsonl` 构建。

只有打标 txt、没有元数据的数据集（如 Kaggle）也能 prepare，此时 booru 元数据列
（rating 等）为空，`--ratings` 过滤不生效。

`metadata.jsonl` 构建时默认取 tag_string_general + tag_string_copyright，
去掉 artist 和 meta 标签（画风训练不应绑定画师名，meta 标签是画质标记）；
`--keep-artist-tag` 可保留画师标签。kohya 的 caption 是纯文本，
不做 WebUI 式括号转义。

## 目录结构

```text
<dataset>/
├─ *.jpg / *.png / *.webp     # 原始图片（只读，不修改）
├─ train/<repeats>_<concept>/ # kohya 结构: 图片(硬链接) + 同名 .txt caption
├─ val/                       # --val-ratio > 0 时的验证集副本（kohya 看不到）
└─ manifest/prepare.parquet
```

图片以硬链接引用根目录原件（同一卷上不占额外空间）；文件系统不支持硬链接
（exFAT、跨卷）时回退为完整复制。

## 验证集切分（--val-ratio）

`--val-ratio 0.1` 时按 image_id（内容哈希）**确定性**切出验证集：同一图片无论重跑
多少次、扩库后归属都稳定；val 图硬链接到 `<dataset>/val/`，kohya 看不到、不参与训练，
供评估（第 8 步）做泛化检测，同时回填 scan 清单的 `split` 列。

> ⚠️ 已完成训练的实验不要对其数据集事后重跑 prepare 加 val——val 抽走后训练集会变少，
> 与已训练的 checkpoint 不再对齐。验证集只惠及之后的实验。
