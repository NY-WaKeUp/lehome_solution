#!/usr/bin/env python3
"""Bypass: parse AWR train console logs -> TensorBoard events under /root/tf-logs.

Does not touch the training process. Tails log files and writes scalars so
AutoDL AutoPanel TensorBoard (pinned to /root/tf-logs, port 6007) shows curves.

Console metric lines only appear every config.log_interval (default 100). Between
those Step lines we carry forward the last known full metric set onto Progress it
so every train/* curve keeps updating in TB.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

STEP_RE = re.compile(r"Step\s+(\d+):\s*(.*)")
KV_RE = re.compile(r"([A-Za-z0-9_./]+)=([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")
PROGRESS_RE = re.compile(r"Progress on:\s+(\d+(?:\.\d+)?)it/")
TIMING_RE = re.compile(
    r"TIMING \[last 10 steps\]:\s*gpu=([0-9.]+)s,\s*data=([0-9.]+)s,\s*total=([0-9.]+)s/step"
)

# Prefer dense progress-driven metric points; Step lines still win when present.
PROGRESS_METRIC_EVERY = 1
PROGRESS_IT_EVERY = 1


def parse_step_line(line: str):
    m = STEP_RE.search(line)
    if not m:
        return None
    step = int(m.group(1))
    metrics = {k: float(v) for k, v in KV_RE.findall(m.group(2))}
    return step, metrics


def write_metrics(writer: SummaryWriter, step: int, metrics: dict, prefix: str = "train"):
    for k, v in metrics.items():
        writer.add_scalar(f"{prefix}/{k}", v, step)
    if "loss" in metrics:
        writer.add_scalar("loss", metrics["loss"], step)


def ingest_file(
    path: Path,
    writer: SummaryWriter,
    state: dict,
    runtime: dict,
    max_step: int | None,
) -> tuple[int, int]:
    """Ingest new bytes from one log file.

    runtime keys:
      last_progress[path] -> last progress it written for progress/it
      last_metric_prog[path] -> last progress it used for carry-forward metrics
      last_metrics -> shared last Step metrics dict
      last_it -> shared latest progress it (for TIMING x-axis)
      written_metric_steps -> set of steps that already got a real/carried metric write
    """
    written = 0
    seen_max = -1
    key = str(path)
    offset = state.get(key, 0)
    if not path.exists():
        return 0, seen_max
    size = path.stat().st_size
    if size < offset:
        offset = 0
    last_progress: dict = runtime["last_progress"]
    last_metric_prog: dict = runtime["last_metric_prog"]
    written_metric_steps: set = runtime["written_metric_steps"]

    with path.open("r", errors="replace") as f:
        f.seek(offset)
        for line in f:
            step_parsed = parse_step_line(line)
            if step_parsed:
                step, metrics = step_parsed
                if max_step is not None and step > max_step:
                    continue
                seen_max = max(seen_max, step)
                if metrics:
                    runtime["last_metrics"] = dict(metrics)
                    runtime["last_it"] = max(runtime.get("last_it", -1), step)
                    write_metrics(writer, step, metrics)
                    written_metric_steps.add(step)
                    written += 1
                continue

            pm = PROGRESS_RE.search(line)
            if pm:
                step = int(float(pm.group(1)))
                if max_step is not None and step > max_step:
                    continue
                runtime["last_it"] = max(runtime.get("last_it", -1), step)
                prev_it = last_progress.get(key, -10**9)
                if step - prev_it >= PROGRESS_IT_EVERY:
                    writer.add_scalar("progress/it", step, step)
                    last_progress[key] = step
                    written += 1
                # Carry forward last known full metric set so TB curves keep
                # moving before the next log_interval Step line arrives.
                metrics = runtime.get("last_metrics") or {}
                prev_metric_it = last_metric_prog.get(key, -10**9)
                if (
                    metrics
                    and step - prev_metric_it >= PROGRESS_METRIC_EVERY
                    and step not in written_metric_steps
                ):
                    write_metrics(writer, step, metrics)
                    last_metric_prog[key] = step
                    written_metric_steps.add(step)
                    written += 1
                continue

            tm = TIMING_RE.search(line)
            if tm:
                step = runtime.get("last_it", 0)
                writer.add_scalar("timing/gpu_s", float(tm.group(1)), step)
                writer.add_scalar("timing/data_s", float(tm.group(2)), step)
                writer.add_scalar("timing/total_s_per_step", float(tm.group(3)), step)
                written += 1
        state[key] = f.tell()
    return written, seen_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="/root/tf-logs/sim_rl_awr_sample_b192")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument(
        "--max-step",
        action="append",
        default=[],
        help="Repeatable: max step for corresponding positional log (use -1 for no cap)",
    )
    ap.add_argument("logs", nargs="+")
    args = ap.parse_args()

    caps: list[int | None] = []
    for i, _ in enumerate(args.logs):
        if i < len(args.max_step):
            v = int(args.max_step[i])
            caps.append(None if v < 0 else v)
        else:
            caps.append(None)

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(logdir))
    state: dict = {}
    runtime = {
        "last_progress": {},
        "last_metric_prog": {},
        "last_metrics": {},
        "last_it": -1,
        "written_metric_steps": set(),
    }
    total_writes = 0
    global_max = -1

    print(f"[log_to_tb] logdir={logdir}", flush=True)
    for p, cap in zip(args.logs, caps):
        print(f"[log_to_tb] watch={p} max_step={cap}", flush=True)

    while True:
        batch = 0
        for p, cap in zip(args.logs, caps):
            n, mx = ingest_file(Path(p), writer, state, runtime, cap)
            batch += n
            if mx >= 0:
                global_max = max(global_max, mx)
        if batch:
            writer.flush()
            total_writes += batch
            metric_steps = sorted(runtime["written_metric_steps"])
            last_metrics = runtime.get("last_metrics") or {}
            print(
                "[log_to_tb] wrote %d step-lines; total_writes=%d; max_step=%s; "
                "metric_points=%d; last_loss=%s; n_keys=%d; last_it=%s"
                % (
                    batch,
                    total_writes,
                    global_max,
                    len(metric_steps),
                    last_metrics.get("loss"),
                    len(last_metrics),
                    runtime.get("last_it"),
                ),
                flush=True,
            )
        if args.once:
            break
        time.sleep(args.poll)

    writer.flush()
    writer.close()
    print(
        "[log_to_tb] done. total_writes=%d max_step=%s" % (total_writes, global_max),
        flush=True,
    )


if __name__ == "__main__":
    main()
