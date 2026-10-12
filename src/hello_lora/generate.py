"""样例图生成：用 diffusers 从既有 checkpoint 后置生成评估样例。

需要主环境的 inference 依赖组（torch / diffusers / transformers / accelerate）:
    uv sync --group inference

显存策略（RTX 4050 6GB 级）：默认 sequential offload（逐层搬运）。实测它
反而最快——fp16 UNet 常驻约 5.0GB，放进 6GB 卡（桌面已占 ~0.7GB）会触发
驱动的 sysmem fallback 静默溢出到系统内存，比逐层搬运慢一个数量级；
sequential 只占 ~1.9GB、GPU 满频，768px 实测约 1.6s/步。手动分相搬运
（offload="model"：prompt 编码后文本编码器回 CPU、UNet 常驻）只在显存
≥ 8GB 时有意义。fp8 量化只用于训练侧（kohya），推理侧引入 torchao /
bitsandbytes 收益有限，不做。

底模只在评估入口加载一次，checkpoint 之间用 load_lora_weights /
unload_lora_weights 只换 LoRA 权重；已存在的目标 png 默认跳过（增量可重跑）。

本模块顶层 import torch / diffusers，只能被 eval.py 在函数内惰性 import，
不要在 cli.py 顶层 import。
"""

from __future__ import annotations

from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline

# 生成参数校验值
MAX_GUIDANCE = 30.0


def build_pipeline(
    base_model: Path, device: str = "cuda", dtype: str = "fp16", offload: str = "sequential"
) -> StableDiffusionXLPipeline:
    """从单文件底模构建 SDXL 推理管线。

    offload: sequential = 逐层 offload（6GB 卡的默认选择，实测最快）；
             model = 手动分相搬运（组件留在 CPU，需依次调用
             encode_prompts / stage_unet，显存 ≥ 8GB 时用）；
             none = 整体驻留显存（大显存卡用）。
    """
    torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype]
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(base_model), torch_dtype=torch_dtype
    )
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    if offload == "model":
        pass  # 组件留在 CPU，由 encode_prompts / stage_unet 分相搬运
    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif offload == "none":
        pipe.to(device)
    else:
        raise SystemExit(f"未知 offload 策略: {offload}（可选 model / sequential / none）")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def encode_prompts(
    pipe: StableDiffusionXLPipeline,
    prompts: list[str],
    device: str = "cuda",
    offload: str = "sequential",
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]:
    """预编码 prompt（含空串负向分支）为 SDXL 联合 embedding。

    返回 (正向 {text: (embeds, pooled)}, 负向 (embeds, pooled))。取法与
    diffusers 内部 encode_prompt 逐位一致（真机验证过）：两个文本编码器
    都取倒数第二层 hidden state 沿特征维拼接，pooled 取第二个编码器的
    投影输出。embedding 只与文本有关，跨 checkpoint 复用，保证同 seed
    下逐 epoch 样例可比。

    offload="model" 时编码完把文本编码器移回 CPU，为 UNet 腾显存；
    sequential 由 offload 钩子自动搬运，none 时本就驻留，均不做手动移动。
    """
    if offload == "model":
        pipe.text_encoder.to(device)
        pipe.text_encoder_2.to(device)

    def encode_one(text: str) -> tuple[torch.Tensor, torch.Tensor]:
        ids1 = pipe.tokenizer(
            text, padding="max_length", max_length=77, truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        ids2 = pipe.tokenizer_2(
            text, padding="max_length", max_length=77, truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        with torch.inference_mode():
            out1 = pipe.text_encoder(ids1, output_hidden_states=True)
            out2 = pipe.text_encoder_2(ids2, output_hidden_states=True)
            embeds = torch.cat(
                [out1.hidden_states[-2], out2.hidden_states[-2]], dim=-1
            )
            pooled = out2[0]
        return embeds, pooled

    embeds_map = {text: encode_one(text) for text in sorted(set(prompts))}
    negative = encode_one("")

    if offload == "model":
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()
    return embeds_map, negative


def stage_unet(pipe: StableDiffusionXLPipeline, device: str = "cuda", offload: str = "sequential") -> None:
    """分相搬运第二步：把 UNet / VAE 搬到 GPU 常驻（offload="model" 时）。"""
    if offload != "model":
        return
    pipe.unet.to(device)
    pipe.vae.to(device)
    torch.cuda.empty_cache()


def generate_for_checkpoint(
    pipe: StableDiffusionXLPipeline,
    lora_path: Path | None,
    jobs: list[tuple[Path, str, int]],
    embeds: dict[str, tuple[torch.Tensor, torch.Tensor]],
    negative: tuple[torch.Tensor, torch.Tensor],
    resolution: int = 768,
    steps: int = 30,
    guidance: float = 7.0,
    force: bool = False,
    device: str = "cuda",
) -> list[Path]:
    """为一个 checkpoint（或 base 基线）生成样例图，返回实际存在的图路径。

    jobs 元素为 (目标 png 路径, prompt, seed)，prompt 必须已在 embeds 里；
    已存在且未 force 的跳过。lora_path 为 None 时不加载 LoRA（base 基线）。
    """
    if not 0 < guidance <= MAX_GUIDANCE:
        raise SystemExit(f"guidance 超出合理范围 (0, {MAX_GUIDANCE}]: {guidance}")
    todo = [(out, prompt, seed) for out, prompt, seed in jobs if force or not out.is_file()]
    generated = [out for out, _, _ in jobs if out.is_file()]

    # 全部命中缓存时不必加载 / 卸载 LoRA
    if lora_path is not None and todo:
        pipe.load_lora_weights(str(lora_path))
    try:
        for out, prompt, seed in todo:
            out.parent.mkdir(parents=True, exist_ok=True)
            pos_embeds, pos_pooled = embeds[prompt]
            generator = torch.Generator(device=device).manual_seed(seed)
            with torch.inference_mode():
                try:
                    image = pipe(
                        prompt_embeds=pos_embeds,
                        pooled_prompt_embeds=pos_pooled,
                        negative_prompt_embeds=negative[0],
                        negative_pooled_prompt_embeds=negative[1],
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        height=resolution,
                        width=resolution,
                        generator=generator,
                    ).images[0]
                except torch.cuda.OutOfMemoryError as exc:
                    raise SystemExit(
                        "生成时显存不足（CUDA OOM）。请降低 --resolution，"
                        "或改用 --offload sequential（更慢但更省显存）"
                    ) from exc
            image.save(out)
            generated.append(out)
            print(f"样例已生成: {out}")
    finally:
        if lora_path is not None and todo:
            pipe.unload_lora_weights()
    return generated
