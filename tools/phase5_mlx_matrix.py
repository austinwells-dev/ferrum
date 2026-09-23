#!/usr/bin/env python3
"""Repeated native MLX-LM generation runs for Ferrum's exact Phase 5 prompts."""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache


class LastPositionLogits:
    """Keep MLX's normal quantized transformer and compute only the sampled row."""

    def __init__(self, model):
        self.model = model

    def __call__(self, inputs, cache=None):
        hidden = self.model.model(inputs, cache=cache)
        hidden = hidden[:, -1:, :]
        if self.model.args.tie_word_embeddings:
            return self.model.model.embed_tokens.as_linear(hidden)
        return self.model.lm_head(hidden)


def run_once(model, prompt_ids: list[int], max_tokens: int) -> dict:
    cache = make_prompt_cache(model)
    logits_model = LastPositionLogits(model)
    start = time.perf_counter()
    timestamps = []
    generated_ids = []
    stream = generate_step(
        mx.array(prompt_ids),
        logits_model,
        max_tokens=max_tokens,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        prompt_cache=cache,
    )
    for token, _logprobs in stream:
        generated_ids.append(int(token))
        timestamps.append(time.perf_counter())
    elapsed = timestamps[-1] - start
    first_token = timestamps[0] - start
    decode_ms = [
        (right - left) * 1000.0 for left, right in zip(timestamps, timestamps[1:])
    ]
    sorted_decode = sorted(decode_ms)
    median_decode_ms = sorted_decode[len(sorted_decode) // 2] if sorted_decode else None
    return {
        "generation_ms": elapsed * 1000.0,
        "generation_tps": len(generated_ids) / elapsed,
        "first_token_ms": first_token * 1000.0,
        "decode_ms": decode_ms,
        "decode_median_ms": median_decode_ms,
        "decode_tps": 1000.0 / median_decode_ms if median_decode_ms else None,
        "generated_ids": generated_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("ferrum_matrix_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise SystemExit("repeats must be positive")

    ferrum_rows = [
        json.loads(line)
        for line in args.ferrum_matrix_jsonl.read_text().splitlines()
        if line.strip()
    ]
    quantized_rows = [row for row in ferrum_rows if row["model"] != "bf16"]
    cases = {}
    for row in quantized_rows:
        cases.setdefault(row["case"], row)
    if not cases:
        raise SystemExit("Ferrum matrix contains no quantized model rows")

    model, _tokenizer = load(str(args.model_dir))
    for case, ferrum_row in cases.items():
        prompt = ferrum_row["prompt_ids"]
        # Warm model, generated kernels and each case's sequence geometry.
        run_once(model, prompt, min(2, len(ferrum_row["generated_ids"])))

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as output:
        for case, ferrum_row in cases.items():
            reference_rows = [
                row
                for row in quantized_rows
                if row["case"] == case
            ]
            prompt = ferrum_row["prompt_ids"]
            max_tokens = len(ferrum_row["generated_ids"])
            for pair in range(args.repeats):
                metrics = run_once(model, prompt, max_tokens)
                reference = reference_rows[pair % len(reference_rows)]["generated_ids"]
                record = {
                    "case": case,
                    "model": "native-mlx-lm",
                    "pair": pair,
                    "pair_order": "follow-up after Ferrum paired matrix",
                    "runtime": "mlx-lm generate_step with final-row quantized logits",
                    "mlx_lm_version": getattr(mlx_lm, "__version__", "unknown"),
                    "mlx_version": getattr(mx, "__version__", "unknown"),
                    "precision": "affine Q4 weights, F16 activations and KV cache",
                    "prompt_tokens": len(prompt),
                    "generated_tokens": max_tokens,
                    "prompt_ids": prompt,
                    "ferrum_generated_ids": reference,
                    "generated_ids_match_ferrum": metrics["generated_ids"] == reference,
                    **metrics,
                }
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
                mx.clear_cache()


if __name__ == "__main__":
    main()
