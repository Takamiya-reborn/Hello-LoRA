# 7. 调参

调参没有专门命令：一个实验就是 `assets/experiments/<id>/` 下的一个 `config.toml`。
复制上一轮的 config 起新实验，每次只改一组变量，训练后用第 8 步的评估横向比较。

> 🚧 自动化 sweep 工具（批量生成 config、汇总多实验报告）规划中，当前为手动流程。

## 流程

1. 以当前最优实验为基线，复制 config 起新实验：

```bash
mkdir assets/experiments/<新id>
cp assets/experiments/<基线id>/config.toml assets/experiments/<新id>/
```

2. 编辑新 config，**只改一组变量**（学习率 / network_dim / repeats / 触发词策略），
   否则指标差异无法归因；
3. 从仓库根目录训练（见 [06-train.md](06-train.md)）；
4. 两个实验各跑一遍 `hello-lora eval`（见 [08-eval.md](08-eval.md)），
   对比各自的 `manifest/eval.report.json` 推荐 epoch 与逐 epoch 曲线。

实验 id 建议带上变量名（如 `nekoda_style-lr1e4` / `nekoda_style-dim16`），
一眼可归因；也可在实验目录里放一个 `notes.md` 记录本轮动机与结论。

## 跨实验可比的前提

eval 指标是 CLIP 相似度，**只有评估条件完全一致时数字才能横向比较**：

- 同一份 `prompts.txt`（把基线实验的复制到新实验，或统一用 `--prompts` 指向同一文件）；
- 同一 `--clip-model`、`--resolution`、`--steps`、`--guidance`、`--samples-per-prompt`；
- 样例 seed 由 prompt_id + `--base-seed` 确定性派生，以上都一致时
  跨实验的同名样例图逐像素可比。

## 变量与预期信号

调整方向与第 8 步指标的对应关系（指标含义见 [08-eval.md](08-eval.md)）：

| 变量 | 调大 | 调小 |
| --- | --- | --- |
| 学习率 | 收敛快，`gap` 拉开早（过拟合提前）；`clip_t` 后期下跌更快 | 稳但需要更多 epoch 才脱离欠拟合线（`style_sim_train` < base + 0.02） |
| `network_dim` / alpha | 容量大，`nearest_train_sim` 更容易越过 0.90（记忆化） | 容量小，画风欠拟合、样例趋同（`diversity` 变化小） |
| repeats（prepare `--repeats`） | 等效于学习率 × 数据重复，一般与 lr 二选一调 | — |
| 触发词 / `--keep-artist-tag` | 改变 caption 分布，影响绑定强度 | — |

改触发词、repeats 属于数据侧变更，需要**重跑 prepare（tag 侧变更还要重跑 tag `--overwrite`）**，
且会改变训练集内容——与旧实验的 checkpoint 不再对齐，只能新建实验从零训。

## 汇总对比（可选）

各实验的 `eval.parquet` 结构相同（每 checkpoint × prompt × seed 一行），
可用 DuckDB 一次拉平多实验的逐 epoch 曲线：

```bash
uv run python -c "
import duckdb
print(duckdb.sql('''
    SELECT regexp_extract(filename, 'experiments/([^/]+)/', 1) AS experiment,
           epoch,
           avg(style_sim_train) AS style,
           avg(clip_t) AS clip_t
    FROM read_parquet('assets/experiments/*/manifest/eval.parquet', filename=true)
    GROUP BY experiment, epoch ORDER BY experiment, epoch
'''))
"
```

> `read_parquet` 的通配符会混入路径能匹配到的所有实验；
> 对比前确保各实验用的是同一 `--clip-model` 与评估条件（见上文可比性前提）。
> 无验证集的实验 `style_sim_val` 为 null，需要时在 SELECT 里单独带出。
