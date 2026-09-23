"""Summarize paired Phase 5 generation JSONL without mixing model variants."""
import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def milliseconds(duration):
    return duration["secs"] * 1000.0 + duration["nanos"] / 1e6


def median(values):
    return statistics.median(values) if values else None


def summarize_profile(entries):
    operations = defaultdict(lambda: defaultdict(float))
    for profile in entries:
        for name, metrics in profile.items():
            row = operations[name]
            for key in ("calls", "allocation_bytes", "gpu_samples"):
                row[key] += metrics[key]
            for key in ("wall", "allocation", "submission", "synchronized", "gpu"):
                row[f"{key}_ms"] += milliseconds(metrics[key])
    return {name: dict(metrics) for name, metrics in sorted(operations.items())}


def summarize_counters(counters):
    if not counters:
        return {}
    result = {}
    for key in ("gpu", "encode", "wait", "allocation_time"):
        result[f"{key}_ms_mean"] = statistics.mean(milliseconds(row[key]) for row in counters)
    for key in (
        "allocations",
        "allocated_bytes",
        "reused_bytes",
        "command_buffers",
        "completion_waits",
        "dispatches",
    ):
        result[f"{key}_mean"] = statistics.mean(row[key] for row in counters)
    result["transient_peak_bytes_max"] = max(row["transient_peak_bytes"] for row in counters)
    result["arena_capacity_bytes_max"] = max(row["arena_capacity"] for row in counters)
    return result


def summarize_runs(rows):
    metrics = (
        "generation_ms",
        "generation_tps",
        "post_first_token_tps",
        "prefill_ms",
        "prefill_tps",
        "first_token_ms",
        "decode_median_ms",
        "decode_tps",
        "decode_aggregate_tps",
        "sampling_ms",
    )
    result = {key: median([row[key] for row in rows]) for key in metrics}
    result["runs"] = len(rows)
    result["prefill_counters"] = summarize_counters([row["prefill_counters"] for row in rows])
    result["decode_counters"] = summarize_counters(
        [counter for row in rows for counter in row["decode_counters"]]
    )
    result["prefill_profile_sum_over_runs"] = summarize_profile(
        [row["prefill_profile"] for row in rows]
    )
    return result


def main(path):
    matrix = Path(path)
    opener = gzip.open if matrix.suffix == ".gz" else open
    with opener(matrix, "rt", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    groups = defaultdict(list)
    pairs = defaultdict(dict)
    for row in rows:
        groups[(row["case"], row["model"])].append(row)
        pairs[(row["case"], row["pair"])][row["model"]] = row
    models = sorted({row["model"] for row in rows if row["model"] != "bf16"})
    quantized = models[0] if len(models) == 1 else None
    workloads = {}
    cases = sorted({row["case"] for row in rows}, key=lambda value: next(
        i for i, row in enumerate(rows) if row["case"] == value
    ))
    for case in cases:
        variant_runs = {
            model: summarize_runs(groups[(case, model)])
            for model in ("bf16", quantized)
            if model is not None and (case, model) in groups
        }
        paired = []
        for (pair_case, pair_id), run in sorted(pairs.items(), key=lambda item: item[0][1]):
            if pair_case != case or "bf16" not in run or quantized not in run:
                continue
            dense, packed = run["bf16"], run[quantized]
            paired.append({
                "pair": pair_id,
                "order": [dense["pair_order"], packed["pair_order"]],
                "prefill_tps_delta_percent": 100.0 * (packed["prefill_tps"] / dense["prefill_tps"] - 1.0),
                "cached_decode_tps_delta_percent": 100.0 * (packed["decode_tps"] / dense["decode_tps"] - 1.0),
                "generation_tps_delta_percent": 100.0 * (packed["generation_tps"] / dense["generation_tps"] - 1.0),
                "first_token_ms_delta_percent": 100.0 * (packed["first_token_ms"] / dense["first_token_ms"] - 1.0),
                "generated_id_mismatches": abs(
                    len(dense["generated_ids"]) - len(packed["generated_ids"])
                ) + sum(
                    left != right
                    for left, right in zip(dense["generated_ids"], packed["generated_ids"])
                ),
            })
        workloads[case] = {
            "models": variant_runs,
            "paired_comparison": {
                "pairs": paired,
                "median_prefill_tps_delta_percent": median([p["prefill_tps_delta_percent"] for p in paired]),
                "median_cached_decode_tps_delta_percent": median([p["cached_decode_tps_delta_percent"] for p in paired]),
                "median_generation_tps_delta_percent": median([p["generation_tps_delta_percent"] for p in paired]),
                "median_first_token_ms_delta_percent": median([p["first_token_ms_delta_percent"] for p in paired]),
                "exact_generated_id_pairs": sum(p["generated_id_mismatches"] == 0 for p in paired),
            },
        }
    output = {
        "source": str(path),
        "quantized_model": quantized,
        "rows": len(rows),
        "workloads": workloads,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_phase5_matrix MATRIX.jsonl[.gz]")
    main(sys.argv[1])
