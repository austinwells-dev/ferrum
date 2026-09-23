#!/usr/bin/env python3
"""Compare Ferrum logits with mlx-lm for the same pinned MLX checkpoint."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx_lm
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def compare_logits(native: np.ndarray, ferrum: np.ndarray) -> dict:
    if native.shape != ferrum.shape or native.ndim != 2:
        raise SystemExit(f"logit shapes differ: mlx={native.shape}, ferrum={ferrum.shape}")
    absolute_sum = 0.0
    squared_sum = 0.0
    dot = 0.0
    native_norm = 0.0
    ferrum_norm = 0.0
    max_absolute = 0.0
    top1_matches = 0
    top5_overlap = 0
    for native_row, ferrum_row in zip(native, ferrum):
        delta = ferrum_row - native_row
        absolute = np.abs(delta)
        absolute_sum += float(np.sum(absolute, dtype=np.float64))
        squared_sum += float(np.sum(delta.astype(np.float64) ** 2, dtype=np.float64))
        max_absolute = max(max_absolute, float(np.max(absolute)))
        native64 = native_row.astype(np.float64)
        ferrum64 = ferrum_row.astype(np.float64)
        dot += float(np.dot(native64, ferrum64))
        native_norm += float(np.dot(native64, native64))
        ferrum_norm += float(np.dot(ferrum64, ferrum64))
        top1_matches += int(np.argmax(native_row) == np.argmax(ferrum_row))
        native_top5 = np.argpartition(native_row, -5)[-5:]
        ferrum_top5 = np.argpartition(ferrum_row, -5)[-5:]
        top5_overlap += len(set(native_top5.tolist()) & set(ferrum_top5.tolist()))

    count = native.size
    positions = native.shape[0]
    return {
        "positions": positions,
        "logit_max_abs_error": max_absolute,
        "logit_mean_abs_error": absolute_sum / count,
        "logit_rmse": float(np.sqrt(squared_sum / count)),
        "logit_cosine_similarity": dot / (np.sqrt(native_norm) * np.sqrt(ferrum_norm)),
        "top1_position_agreement": top1_matches / positions,
        "top5_mean_overlap": top5_overlap / (positions * 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("ferrum_quality_json", type=Path)
    parser.add_argument("ferrum_prefill_logits_f32le", type=Path)
    parser.add_argument("ferrum_decode_logits_f32le", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.ferrum_quality_json.read_text())
    model, tokenizer = load(str(args.model_dir))
    ids = tokenizer.encode(manifest["corpus"], add_special_tokens=False)
    if ids != manifest["token_ids"]:
        raise SystemExit("MLX and Ferrum tokenizers produced different IDs")

    native_logits = model(mx.array([ids]))
    mx.eval(native_logits)
    vocabulary = int(manifest["vocabulary"])
    native = np.asarray(native_logits).astype(np.float32, copy=False).reshape(
        len(ids), vocabulary
    )
    ferrum = np.fromfile(args.ferrum_prefill_logits_f32le, dtype="<f4").reshape(
        len(ids), vocabulary
    )

    native_cache = make_prompt_cache(model)
    native_decode_rows = []
    for token in ids[:-1]:
        logits = model(mx.array([[token]]), cache=native_cache)
        mx.eval(logits)
        native_decode_rows.append(
            np.asarray(logits[0, -1]).astype(np.float32, copy=True)
        )
    native_decode = np.stack(native_decode_rows)
    ferrum_decode = np.fromfile(args.ferrum_decode_logits_f32le, dtype="<f4").reshape(
        len(ids) - 1, vocabulary
    )

    result = {
        "comparison": "Ferrum direct Metal affine-Q4 versus native mlx-lm quantized forward",
        "mlx_lm_version": getattr(mlx_lm, "__version__", "unknown"),
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "repository": manifest["quantized_repository"],
        "revision": manifest["quantized_revision"],
        "vocabulary": vocabulary,
        "token_ids_match": True,
        "prefill": compare_logits(native, ferrum),
        "cached_decode": compare_logits(native_decode, ferrum_decode),
        "native_dtype": str(native_logits.dtype),
        "ferrum_export_dtype": "f32 (converted from Ferrum F16 output)",
        "ferrum_quality_manifest": str(args.ferrum_quality_json),
    }
    args.output_json.write_text(json.dumps(result, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
