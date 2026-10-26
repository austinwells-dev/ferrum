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

With --image (repeatable) the run is an edit: the condition images are
resized as the pipeline does (to about --resolution squared, sides multiples
of 32), the output size follows the last one, and these are added:
  condition.<i> [H, W, 4]      the resized RGBA pixels, 0..255
  vision_embeds [V, 4096]      vision tower output for all images (merged rows)
  deepstack.<k> [V, 4096]      DeepStack features of all images
  vision_fp32, deepstack_fp32.<k>  the same from the tower run in float32
  image_pad_mask [T]           1 where the encoder's sequence holds an image slot
  condition_latents [C, 64]    normalised condition latents entering the DiT
  vae_encoded.<i> [h*w, 64]    the same from a float32 VAE

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

import numpy as np
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
    ap.add_argument("--image", action="append", default=[], help="condition image (an edit); repeatable")
    ap.add_argument("--resolution", type=int, default=512, help="edit: target side the images are sized to")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from diffusers import QwenImage21Pipeline
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
    from PIL import Image
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
    f32 = lambda t: t.detach().float().cpu().contiguous()

    # Condition images, resized once as the pipeline does for both encoders.
    conditions = []
    for path in args.image:
        img = Image.open(path).convert("RGBA")
        w_, h_, _ = calculate_dimensions(args.resolution**2, img.size[0] / img.size[1])
        conditions.append(pipe.image_processor.resize(img, width=w_, height=h_))
    if conditions:
        args.width, args.height, _ = calculate_dimensions(
            args.resolution**2, conditions[-1].size[0] / conditions[-1].size[1]
        )
        for i, img in enumerate(conditions):
            out[f"condition.{i}"] = torch.from_numpy(np.asarray(img, dtype=np.float32).copy())

    seen = {}

    def on_encoder(_mod, _a, kw):
        seen["ids"] = kw["input_ids"][0].tolist()

    def on_visual(_mod, _a, _kw, res):
        out["vision_embeds"] = f32(res.pooler_output)
        for k, feature in enumerate(res.deepstack_features):
            out[f"deepstack.{k}"] = f32(feature)

    encoder.register_forward_pre_hook(on_encoder, with_kwargs=True)
    visual_hook = encoder.model.visual.register_forward_hook(on_visual, with_kwargs=True)
    embeds, mask, pad_mask = pipe.encode_prompt(args.prompt, image=conditions or None, device=dev)
    if mask is not None:
        sys.exit("unexpected padding for a single prompt")
    ids = seen["ids"]
    out["prompt_embeds"] = embeds[0].float().cpu().contiguous()
    if conditions:
        out["image_pad_mask"] = f32(pad_mask[0])
        # The tower again in float32 (its weights are BF16 either way), so
        # the vision check is not bounded by the pipeline's BF16 noise.
        visual_hook.remove()
        visual = encoder.model.visual.to("cpu", torch.float32)
        whites = []
        for img in conditions:
            white = Image.new("RGB", img.size, (255, 255, 255))
            white.paste(img, mask=img.getchannel("A"))
            whites.append(white)
        pixels = pipe.processor.image_processor(images=whites, return_tensors="pt")
        with torch.no_grad():
            res = visual(pixels.pixel_values.float(), grid_thw=pixels.image_grid_thw)
        out["vision_fp32"] = f32(res.pooler_output).clone()
        for k, feature in enumerate(res.deepstack_features):
            out[f"deepstack_fp32.{k}"] = f32(feature).clone()
        del visual
    meta = {
        "model": args.model,
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "dtype": args.dtype,
        "input_ids": ids,
        "images": [{"path": p, "width": c.size[0], "height": c.size[1]} for p, c in zip(args.image, conditions)],
        "resolution": args.resolution,
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
    if conditions:
        # The VAE encodes the condition images inside the call, and the call
        # cannot take precomputed embeddings with image slots, so it gets them
        # back from here.
        pipe.vae.to(dev)
        encoded = (embeds, None, pad_mask)
        pipe.encode_prompt = lambda *a, **kw: tuple(t.to(dev) if t is not None else None for t in encoded)

    h, w = args.height // 16, args.width // 16
    gen = torch.Generator("cpu").manual_seed(args.seed)
    noise = torch.randn((1, 64, 1, h, w), generator=gen, dtype=torch.float32)
    noise = noise.view(1, 64, h * w).transpose(1, 2).contiguous()
    out["noise"] = noise[0].clone()

    calls = {"n": 0}

    def on_model_pre(_mod, _args, _kw):
        calls["n"] += 1

    def on_model_post(mod, a, kw, res):
        step = calls["n"] - 1
        sample = res[0] if isinstance(res, tuple) else res.sample
        out[f"pred.{step}"] = f32(sample[0, -h * w :])

    transformer.register_forward_pre_hook(on_model_pre, with_kwargs=True)
    transformer.register_forward_hook(on_model_post, with_kwargs=True)

    def on_first_call(mod, a, kw):
        if calls["n"] == 1 and conditions:
            out["condition_latents"] = f32(kw["hidden_states"][0, : -h * w])

    transformer.register_forward_pre_hook(on_first_call, with_kwargs=True)

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
        prompt=args.prompt if conditions else None,
        prompt_embeds=None if conditions else embeds.to(dev),
        image=conditions or None,
        output_resolution=args.resolution,
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
    for i, img in enumerate(conditions):
        pixels = pipe.image_processor.preprocess(img, width=img.size[0], height=img.size[1]).unsqueeze(2)
        with torch.no_grad():
            moments = vae.encode(pixels.to(dev, torch.float32)).latent_dist.mode()
        z = (moments - mean) / std
        out[f"vae_encoded.{i}"] = f32(z[0, :, 0].reshape(64, -1).T)
    pipe.image_processor.postprocess(image.cpu(), output_type="pil")[0].save(os.path.join(args.out, "reference.png"))

    save_file(out, os.path.join(args.out, "reference.safetensors"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", args.out, {k: list(v.shape) for k, v in out.items() if k in ("prompt_embeds", "noise", "image")})


if __name__ == "__main__":
    main()
