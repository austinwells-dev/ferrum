#!/usr/bin/env python3
"""Wait before Phase 7A benchmarks when another process is using CPU."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessSample:
    pid: int
    cpu_seconds: float
    command: str


def _cpu_seconds(value: str) -> float:
    """Parse the cumulative TIME field printed by macOS ps."""
    parts = value.strip().split(":")
    try:
        seconds = float(parts[-1])
        if len(parts) == 1:
            return seconds
        minutes = int(parts[-2])
        if len(parts) == 2:
            return minutes * 60 + seconds
        hours = int(parts[-3])
        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, IndexError) as error:
        raise ValueError(f"cannot parse ps CPU time {value!r}") from error


def _ignored() -> set[str]:
    """Executable basenames excluded via PHASE7A_PREFLIGHT_IGNORE (comma-separated).

    Opt-in only, for a stuck system daemon during internal paired A/Bs; every
    exemption is echoed on the preflight line so affected runs are identifiable.
    """
    return {name.strip() for name in os.environ.get("PHASE7A_PREFLIGHT_IGNORE", "").split(",") if name.strip()}


def _snapshot() -> dict[int, ProcessSample]:
    result = subprocess.run(
        ["ps", "-A", "-o", "pid=,time=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    samples: dict[int, ProcessSample] = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3:
            continue
        try:
            pid = int(fields[0])
            cpu_seconds = _cpu_seconds(fields[1])
        except ValueError:
            continue
        executable = fields[2].split()[0].rsplit("/", 1)[-1] if fields[2] else ""
        if pid != os.getpid() and executable not in _ignored():
            samples[pid] = ProcessSample(pid, cpu_seconds, fields[2])
    return samples


def _measure(samples: int, interval: float) -> tuple[float, list[tuple[float, str]]]:
    previous = _snapshot()
    peak_total = 0.0
    peak_processes: dict[int, tuple[float, str]] = {}
    for _ in range(samples - 1):
        started = time.monotonic()
        time.sleep(interval)
        elapsed = time.monotonic() - started
        current = _snapshot()
        total = 0.0
        for pid, now in current.items():
            before = previous.get(pid)
            if before is None:
                continue
            used = max(0.0, now.cpu_seconds - before.cpu_seconds)
            percent = used * 100.0 / elapsed
            total += percent
            if percent > peak_processes.get(pid, (0.0, ""))[0]:
                peak_processes[pid] = (percent, now.command)
        peak_total = max(peak_total, total)
        previous = current
    busiest = sorted(peak_processes.values(), reverse=True)[:5]
    return peak_total, busiest


def _check(args: argparse.Namespace) -> bool:
    total, busiest = _measure(args.samples, args.interval)
    busy_processes = [
        (percent, command)
        for percent, command in busiest
        if percent > args.max_process_percent
    ]
    busy = total > args.max_total_core_percent or bool(busy_processes)
    if busy:
        print(
            "CPU preflight BLOCKED: competing CPU load exceeds the benchmark limit "
            f"(total {total:.1f}% of one core; limit {args.max_total_core_percent:.1f}%).",
            file=sys.stderr,
        )
        for percent, command in busiest:
            print(f"  {percent:5.1f}% of one core  {command[:180]}", file=sys.stderr)
        return False
    ignored = sorted(_ignored())
    if ignored:
        print(f"CPU preflight exempting: {', '.join(ignored)}", flush=True)
    print(
        "CPU preflight passed: "
        f"peak total {total:.1f}% of one core; "
        f"largest process {busiest[0][0]:.1f}% of one core."
        if busiest
        else f"CPU preflight passed: peak total {total:.1f}% of one core.",
        flush=True,
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sample process CPU-time deltas before benchmarking. By default this "
            "waits above 80% of one core for any process or 220% aggregate CPU, "
            "measured over three samples."
        )
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--max-process-percent", type=float, default=80.0)
    parser.add_argument("--max-total-core-percent", type=float, default=220.0)
    parser.add_argument("--wait", action="store_true", help="retry until the host is clear")
    parser.add_argument("--retry-seconds", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.samples < 2 or args.interval <= 0:
        parser.error("--samples must be at least 2 and --interval must be positive")
    if args.max_process_percent < 0 or args.max_total_core_percent < 0:
        parser.error("CPU limits cannot be negative")

    while not _check(args):
        if not args.wait:
            return 2
        print(f"Waiting {args.retry_seconds:g}s before checking again.", flush=True)
        time.sleep(args.retry_seconds)

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if command:
        os.execvp(command[0], command)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"CPU preflight failed: {error}", file=sys.stderr)
        raise SystemExit(2)
