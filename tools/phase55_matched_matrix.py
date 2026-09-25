#!/usr/bin/env python3
"""Interleave warmed Ferrum, current llama.cpp, and/or native MLX runs.

The input matrix supplies the exact prompt token IDs and requested generation
lengths. llama.cpp receives those IDs through its C API, bypassing retokenizing.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def source_cases(path: Path) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_jsonl(path)
        if row.get("model", row.get("source_matrix_case_model")) != "bf16"
    ]
    cases: dict[str, dict[str, Any]] = {}
    for row in rows:
        case = str(row["case"])
        request_options = row.get("request_options", {})
        if not isinstance(request_options, dict):
            raise ValueError(f"case {case!r} request_options must be an object")
        requested = row.get("max_new_tokens", len(row.get("generated_ids", [])))
        if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
            raise ValueError(f"case {case!r} needs a positive integer max_new_tokens")
        previous = cases.get(case)
        if previous is not None and (
            previous["prompt_ids"] != row["prompt_ids"]
            or previous.get("max_new_tokens", len(previous.get("generated_ids", []))) != requested
            or previous.get("request_options", {}) != request_options
        ):
            raise ValueError(f"case {case!r} has inconsistent prompt, output length, or request options")
        cases.setdefault(case, row)
    if not cases:
        raise ValueError("input matrix has no quantized workload rows")
    return list(cases.values())


def case_request_options(row: dict[str, Any]) -> dict[str, Any]:
    options = row.get("request_options", {})
    if not isinstance(options, dict):
        raise ValueError(f"case {row.get('case')!r} request_options must be an object")
    reserved = {"case", "pair", "pair_order", "warmup", "prompt_ids", "max_new_tokens"}
    collisions = reserved.intersection(options)
    if collisions:
        raise ValueError(
            f"case {row.get('case')!r} request_options override reserved fields: {sorted(collisions)}"
        )
    return options


def process_rss_bytes(pid: int) -> int | None:
    try:
        output = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return int(output) * 1024 if output else None
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


class JsonLineProcess:
    def __init__(self, command: list[str], env: dict[str, str] | None = None):
        self.launched_at = time.perf_counter()
        self.startup_ms: float | None = None
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            status = self.process.poll()
            raise RuntimeError(f"benchmark process ended without a response (status {status})")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        if self.startup_ms is None:
            self.startup_ms = (time.perf_counter() - self.launched_at) * 1000.0
        response["process_rss_bytes"] = process_rss_bytes(self.process.pid)
        return response

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        status = self.process.wait(timeout=60)
        if status != 0:
            raise RuntimeError(f"benchmark process exited with status {status}")


class FerrumVariant:
    """One per-request Ferrum option variant backed by a shared warm process."""

    def __init__(self, process: JsonLineProcess, field: str, value: Any, name: str):
        self.process = process
        self.field = field
        self.value = value
        self.name = name
        self.startup_ms: float | None = None

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = dict(payload)
        request[self.field] = self.value
        response = self.process.request(request)
        self.startup_ms = self.process.startup_ms
        response["ferrum_variant"] = self.name
        return response


class NativeMlx:
    def __init__(self, model_dir: Path):
        self.started_at = time.perf_counter()
        import mlx.core as mx
        import mlx_lm
        from mlx_lm import load
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache

        self.mx = mx
        self.mlx_lm = mlx_lm
        self.generate_step = generate_step
        self.make_prompt_cache = make_prompt_cache
        self.model, _tokenizer = load(str(model_dir))
        self.model_dir = model_dir
        self.model_load_ms = (time.perf_counter() - self.started_at) * 1000.0
        self.startup_ms: float | None = None

    @staticmethod
    def _cache_bytes(cache: Any) -> int:
        total = 0
        for layer in cache:
            state = getattr(layer, "state", None)
            if state is None:
                continue
            values = state if isinstance(state, (tuple, list)) else (state,)
            for value in values:
                if hasattr(value, "nbytes"):
                    total += int(value.nbytes)
        return total

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        mx = self.mx
        cache = self.make_prompt_cache(self.model)

        class LastPositionLogits:
            def __init__(self, model: Any):
                self.model = model

            def __call__(self, inputs: Any, cache: Any = None) -> Any:
                hidden = self.model.model(inputs, cache=cache)
                hidden = hidden[:, -1:, :]
                if self.model.args.tie_word_embeddings:
                    return self.model.model.embed_tokens.as_linear(hidden)
                return self.model.lm_head(hidden)

        prompt = request["prompt_ids"]
        max_tokens = request["max_new_tokens"]
        timestamps: list[float] = []
        prefill_ready: list[float] = []
        mx.reset_peak_memory()
        started = time.perf_counter()

        def prompt_progress(processed: int, total: int) -> None:
            if 0 < processed < total:
                prefill_ready.append(time.perf_counter())

        stream = self.generate_step(
            mx.array(prompt, dtype=mx.uint32),
            LastPositionLogits(self.model),
            max_tokens=max_tokens,
            sampler=lambda logits: mx.argmax(logits, axis=-1),
            prompt_cache=cache,
            prompt_progress_callback=prompt_progress,
        )
        generated_ids: list[int] = []
        for token, _logprobs in stream:
            generated_ids.append(int(token))
            timestamps.append(time.perf_counter())

        elapsed = (timestamps[-1] - started) if timestamps else 0.0
        # mlx_lm.generate_step schedules one look-ahead token before yielding each
        # token. The final scheduled token is not part of this request's output;
        # wait for it after capturing the requested-output boundary so its work
        # cannot spill into the next paired sample.
        mx.synchronize()
        first_token_ms = (timestamps[0] - started) * 1000.0 if timestamps else 0.0
        prefill_ms = (prefill_ready[0] - started) * 1000.0 if prefill_ready else 0.0
        prefill_tokens = max(0, len(prompt) - 1) if prefill_ready else 0
        intervals = [
            (right - left) * 1000.0
            for left, right in zip(timestamps, timestamps[1:])
        ]
        sorted_intervals = sorted(intervals)
        median_decode_ms = sorted_intervals[len(sorted_intervals) // 2] if sorted_intervals else None
        cache_bytes = self._cache_bytes(cache)
        active_memory = int(mx.get_active_memory())
        peak_memory = int(mx.get_peak_memory())
        mx.clear_cache()

        result = {
            "runtime": "native-mlx-lm",
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_lm_version": importlib.metadata.version("mlx-lm"),
            "case": request["case"],
            "pair": request["pair"],
            "pair_order": request["pair_order"],
            "warmup": request.get("warmup", False),
            "model_format": "affine Q4 group-size 64",
            "prompt_tokens": len(prompt),
            "generated_tokens": len(generated_ids),
            "prompt_ids": prompt,
            "generated_ids": generated_ids,
            "generation_ms": elapsed * 1000.0,
            "generation_tps": len(generated_ids) / elapsed if elapsed else 0.0,
            "prefill_tokens": prefill_tokens,
            "prefill_ms": prefill_ms,
            "prefill_tps": prefill_tokens * 1000.0 / prefill_ms if prefill_ms else None,
            "last_prompt_token_and_sample_ms": max(0.0, first_token_ms - prefill_ms),
            "first_token_ms": first_token_ms,
            "decode_ms": intervals,
            "decode_median_ms": median_decode_ms,
            "decode_tps": 1000.0 / median_decode_ms if median_decode_ms else None,
            "decode_aggregate_tps": (
                len(intervals) * 1000.0 / sum(intervals)
                if intervals and sum(intervals) > 0.0
                else None
            ),
            "sampling_ms": None,
            "kv_active_bytes": cache_bytes,
            "mlx_active_memory_bytes": active_memory,
            "mlx_peak_memory_bytes": peak_memory,
            "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "prefill_step_size": 2048,
            "generation_policy": "mlx_lm.generate_step; token IDs supplied directly; greedy argmax",
            "model_load_ms": self.model_load_ms,
        }
        if request.get("warmup") and self.startup_ms is None:
            self.startup_ms = (time.perf_counter() - self.started_at) * 1000.0
        return result


def pair_order(names: list[str], case_index: int, pair: int) -> list[str]:
    offset = case_index % len(names)
    rotated = names[offset:] + names[:offset]
    return rotated if pair % 2 == 0 else list(reversed(rotated))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path, help="Ferrum matrix JSONL or JSONL.GZ")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ferrum-exe", type=Path)
    parser.add_argument("--ferrum-model", type=Path)
    parser.add_argument("--llama-exe", type=Path)
    parser.add_argument("--llama-model", type=Path)
    parser.add_argument("--mlx-model", type=Path)
    parser.add_argument("--profile-ferrum", action="store_true")
    parser.add_argument(
        "--ferrum-paired-option",
        help="interleave two Ferrum values in one process as FIELD=CONTROL,CANDIDATE (values are JSON)",
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if bool(args.ferrum_exe) != bool(args.ferrum_model):
        raise SystemExit("--ferrum-exe and --ferrum-model must be supplied together")
    if bool(args.llama_exe) != bool(args.llama_model):
        raise SystemExit("--llama-exe and --llama-model must be supplied together")
    if args.mlx_model and args.llama_exe:
        raise SystemExit("run GGUF/llama.cpp and MLX affine-Q4 comparisons as separate matched workloads")
    if args.ferrum_model and args.llama_model and args.ferrum_model.resolve() != args.llama_model.resolve():
        raise SystemExit("Ferrum and llama.cpp must receive the same GGUF artifact")
    if args.ferrum_model and args.mlx_model and args.ferrum_model.resolve() != args.mlx_model.resolve():
        raise SystemExit("Ferrum and native MLX must receive the same affine-Q4 model directory")
    if args.ferrum_paired_option and not args.ferrum_exe:
        raise SystemExit("--ferrum-paired-option requires --ferrum-exe and --ferrum-model")

    paired_option: tuple[str, Any, Any] | None = None
    if args.ferrum_paired_option:
        field, separator, values = args.ferrum_paired_option.partition("=")
        control, value_separator, candidate = values.partition(",")
        if not separator or not value_separator or not field:
            raise SystemExit("--ferrum-paired-option must be FIELD=CONTROL,CANDIDATE")
        try:
            paired_option = (field, json.loads(control), json.loads(candidate))
        except json.JSONDecodeError as error:
            raise SystemExit(f"paired Ferrum option values must be JSON: {error}") from error

    engines: dict[str, Any] = {}
    mlx = NativeMlx(args.mlx_model) if args.mlx_model else None
    if mlx is not None:
        engines["native-mlx-lm"] = mlx

    if args.ferrum_exe:
        ferrum_env = os.environ.copy()
        if args.profile_ferrum:
            ferrum_env["FERRUM_MATRIX_PROFILE"] = "1"
        ferrum_process = JsonLineProcess([str(args.ferrum_exe), str(args.ferrum_model)], ferrum_env)
        if paired_option is None:
            engines["ferrum"] = ferrum_process
        else:
            field, control, candidate = paired_option
            engines["ferrum-control"] = FerrumVariant(ferrum_process, field, control, "control")
            engines["ferrum-candidate"] = FerrumVariant(ferrum_process, field, candidate, "candidate")
    if args.llama_exe:
        engines["llama.cpp"] = JsonLineProcess(
            [str(args.llama_exe), str(args.llama_model)], os.environ.copy()
        )
    if not engines:
        raise SystemExit("select at least one of --ferrum-exe, --llama-exe, or --mlx-model")

    cases = source_cases(args.cases)
    names = list(engines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Warm each model, sequence length, and decode path before recording paired runs.
        for case_index, row in enumerate(cases):
            warm_request = {
                "case": row["case"],
                "pair": 0,
                "pair_order": "warmup",
                "warmup": True,
                "prompt_ids": row["prompt_ids"],
                "max_new_tokens": 2,
            }
            warm_request.update(case_request_options(row))
            for engine_name in pair_order(names, case_index, 0):
                engine = engines[engine_name]
                if engine_name == "native-mlx-lm":
                    engine.run(warm_request)
                else:
                    engine.request(warm_request)

        with args.output.open("w", encoding="utf-8") as output:
            for case_index, row in enumerate(cases):
                prompt = row["prompt_ids"]
                max_tokens = row.get("max_new_tokens", len(row.get("generated_ids", [])))
                for pair in range(args.repeats):
                    order = pair_order(names, case_index, pair)
                    order_label = ">".join(order)
                    request = {
                        "case": row["case"],
                        "pair": pair,
                        "pair_order": order_label,
                        "warmup": False,
                        "prompt_ids": prompt,
                        "max_new_tokens": max_tokens,
                    }
                    request.update(case_request_options(row))
                    for engine_name in order:
                        engine = engines[engine_name]
                        result = (
                            engine.run(request)
                            if engine_name == "native-mlx-lm"
                            else engine.request(request)
                        )
                        result["source_matrix_case_model"] = row.get(
                            "model", row.get("source_matrix_case_model", "unknown")
                        )
                        if engine_name == "ferrum":
                            result["ferrum_options"] = {
                                "batch_limit_dispatches": int(os.environ.get("FERRUM_BATCH_LIMIT", "1024")),
                                "q4_0_gemv_8rows": os.environ.get("FERRUM_Q4_0_GEMV_8ROWS", "true").lower() in ("1", "true"),
                                "q4_k_gemv_8rows": os.environ.get("FERRUM_Q4_K_GEMV_8ROWS", "true").lower() in ("1", "true"),
                                "q4_k_mpp_tile_m128": result.get("q4_k_mpp_tile_m128", True),
                                "q4_k_expert_project_8rows": result.get("q4_k_expert_project_8rows", False),
                                "q6_k_expert_project_8rows": result.get("q6_k_expert_project_8rows", False),
                                "attention_softmax_prefix_reuse": result.get("attention_softmax_prefix_reuse", False),
                                "q6_k_gemv_8rows": os.environ.get("FERRUM_Q6_K_GEMV_8ROWS", "true").lower() in ("1", "true"),
                                "moe_gpu_routing": os.environ.get("FERRUM_MOE_GPU_ROUTING", "true").lower() in ("1", "true"),
                                "mlx_affine4_gemv_quad": os.environ.get("FERRUM_MLX_AFFINE4_GEMV_QUAD", "true").lower() in ("1", "true"),
                            }
                        result["engine_order"] = order
                        result["case_index"] = case_index
                        result["engine_load_plus_first_warmup_ms"] = engines[engine_name].startup_ms
                        output.write(json.dumps(result, separators=(",", ":")) + "\n")
                        output.flush()
                        print(json.dumps(result, separators=(",", ":")), flush=True)
    finally:
        closed: set[int] = set()
        for engine in engines.values():
            process = engine.process if isinstance(engine, FerrumVariant) else engine
            if isinstance(process, JsonLineProcess) and id(process) not in closed:
                process.close()
                closed.add(id(process))


if __name__ == "__main__":
    main()
