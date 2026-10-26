#!/usr/bin/env python3
"""Run Qwen-Image-2.1(-Turbo) under diffusers and dump what Ferrum's image
pipeline is checked against.

The three models (text encoder, transformer, VAE) are loaded one after the
other and freed in between, so the run fits a 32 GB Mac. Noise comes from a
seeded torch generator and is handed to the pipeline as `latents`, so Ferrum
starts from the same bytes instead of reproducing torch's RNG.

Output, in --out:
  reference.safetensors   float32 tensors (names below)
  meta.json               prompt, size, seed, token ids, sigmas, dtypes
  reference.png           the decoded image

Tensors (T = prompt tokens after the system prompt, N = latent tokens):
  prompt_embeds [T, 4096]      encoder layer-36 output, before its final norm
  noise [N, 64]                initial packed latents
  rope_cos, rope_sin [T+N, 64] rotary tables for the joint sequence
  modulation [2, 16384]        shared modulation: row 0 sampled t, row 1 t = 0
  block_in [T+N, 4096]         joint sequence entering block 0 (step 0)
  block_out.<i> [T+N, 4096]    block i output, step 0 (prefill)
  cached_out.<i> [N, 4096]     block i output, step 1 (KV-cached, target rows)
  pred.<s> [N, 64]             transformer output at step s
  latents.<s> [N, 64]          latents after step s
  vae_in [64, h, w]            de-normalised latents handed to the VAE
  image [3, H, W]              VAE output in [-1, 1]
"""

import argparse
import gc
import json
import os
import sys

import torch
from safetensors.torch import save_file


def cleanup(device):
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen-Image-2.1-Turbo", help="repo id or local directory")
    ap.add_argument("--prompt", default="A red fox sitting in fresh snow, golden hour light.")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--blocks", default="0,1,15,31", help="comma-separated block indices to dump")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from diffusers import QwenImage21Pipeline
    from transformers import Qwen3VLForConditionalGeneration
    from diffusers import QwenImage21Transformer2DModel

    dtype = getattr(torch, args.dtype)
    dev = args.device
    os.makedirs(args.out, exist_ok=True)
    out = {}
    blocks = [int(b) for b in args.blocks.split(",") if b]

    # Stage 1: text encoder. The pipeline is built without the transformer so the
    # two never share memory.
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, subfolder="text_encoder", dtype=dtype
    ).to(dev)
    pipe = QwenImage21Pipeline.from_pretrained(
        args.model, text_encoder=encoder, transformer=None, dtype=dtype
    )
    pipe.vae.to("cpu")
    embeds, mask, _ = pipe.encode_prompt(args.prompt, device=dev)
    if mask is not None:
        sys.exit("unexpected padding for a single prompt")
    ids = pipe.processor(text=[pipe.prompt_template_t2i.format(args.prompt)], return_tensors="pt").input_ids
    out["prompt_embeds"] = embeds[0].float().cpu().contiguous()
    meta = {
        "model": args.model,
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "dtype": args.dtype,
        "input_ids": ids[0].tolist(),
        "drop_idx": pipe._drop_idx,
        "sigmas": list(pipe.config.sample_sigmas),
    }
    embeds = embeds.to("cpu")
    pipe.text_encoder = None
    del encoder
    cleanup(dev)

    # Stage 2: transformer, with hooks on the first two calls (prefill, cached).
    transformer = QwenImage21Transformer2DModel.from_pretrained(
        args.model, subfolder="transformer", dtype=dtype
    ).to(dev)
    pipe.transformer = transformer

    h, w = args.height // 16, args.width // 16
    gen = torch.Generator("cpu").manual_seed(args.seed)
    noise = torch.randn((1, 64, 1, h, w), generator=gen, dtype=torch.float32)
    noise = noise.view(1, 64, h * w).transpose(1, 2).contiguous()
    out["noise"] = noise[0].clone()

    calls = {"n": 0}
    f32 = lambda t: t.detach().float().cpu().contiguous()

    def on_model_pre(_mod, _args, _kw):
        calls["n"] += 1

    def on_model_post(mod, a, kw, res):
        step = calls["n"] - 1
        sample = res[0] if isinstance(res, tuple) else res.sample
        out[f"pred.{step}"] = f32(sample[0, -h * w :])

    transformer.register_forward_pre_hook(on_model_pre, with_kwargs=True)
    transformer.register_forward_hook(on_model_post, with_kwargs=True)

    def first_block_pre(mod, a, kw):
        if calls["n"] != 1:
            return
        out["block_in"] = f32(kw["hidden_states"][0])
        out["modulation"] = f32(kw["modulation"])
        rope = kw["rotary_emb"]
        out["rope_cos"], out["rope_sin"] = f32(rope.real), f32(rope.imag)

    transformer.transformer_blocks[0].register_forward_pre_hook(first_block_pre, with_kwargs=True)

    for i in blocks:

        def on_block(mod, a, kw, res, i=i):
            if calls["n"] == 1:
                out[f"block_out.{i}"] = f32(res[0])
            elif calls["n"] == 2:
                out[f"cached_out.{i}"] = f32(res[0])

        transformer.transformer_blocks[i].register_forward_hook(on_block, with_kwargs=True)

    def on_step(pipeline, step, t, kw):
        out[f"latents.{step}"] = f32(kw["latents"][0])
        return kw

    latents = pipe(
        prompt_embeds=embeds.to(dev),
        height=args.height,
        width=args.width,
        latents=noise,
        output_type="latent",
        use_kv_cache=True,
        callback_on_step_end=on_step,
    ).images
    final = latents.float().cpu()
    pipe.transformer = None
    del transformer
    cleanup(dev)

    # Stage 3: VAE in float32 (it is small, and this isolates its error from the DiT's).
    vae = pipe.vae.to(torch.float32).to(dev)
    z = final.to(dev).transpose(1, 2).reshape(1, 64, 1, h, w)
    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, 64, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=dev).view(1, 64, 1, 1, 1)
    z = z * std + mean
    out["vae_in"] = f32(z[0, :, 0])
    with torch.no_grad():
        image = vae.decode(z, return_dict=False)[0][:, :, 0]
    out["image"] = f32(image[0])
    pipe.image_processor.postprocess(image.cpu(), output_type="pil")[0].save(os.path.join(args.out, "reference.png"))

    save_file(out, os.path.join(args.out, "reference.safetensors"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", args.out, {k: list(v.shape) for k, v in out.items() if k in ("prompt_embeds", "noise", "image")})


if __name__ == "__main__":
    main()
