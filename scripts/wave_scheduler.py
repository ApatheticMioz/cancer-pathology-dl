#!/usr/bin/env python3
"""wave_scheduler.py — scaling campaign runner (helper for scripts/run_wave.sh).

Hardware target: WSL, 20 GB RAM, 8 procs, 2 GB swap, RTX 3090 24 GB.

Design (measured, not guessed):
  1. VRAM-aware admission: per-job predicted VRAM from the R3-1 measured
     table (vgg16: tcga 8816, pannuke 8429, siim 6635, panda 3205 MiB)
     scaled by per-encoder factors derived from the 26-config matrix in
     run_all_experiments.sh (vgg16 = 1.0 measured; mobilenet_v2 = 0.45
     derived estimate — 13 of the 26 configs use it, no R3-1 mobilenet
     measurement exists, so the factor is a documented estimate).
     Admission gate: live free VRAM (nvidia-smi) - 1.5 GB safety margin
     >= predicted. Jobs that do not fit are DEFERRED (re-checked every
     cycle), never aborted.
  2. RAM watchdog: per-job RSS sampled every cycle. A job breaching its
     budget (GLOBAL_RAM_SOFT_CAP_GB / max_jobs) is hard-aborted ALONE
     (wave continues). Global soft cap ~17 GB is logged when breached.
  3. CPU saturation: adaptive dataloader workers per job from 8 procs /
     active jobs (1->6, 2->3, 3-4->2, 5->1). Requested via --num-workers;
     persistent_workers/prefetch_factor follow src/loader_tuning.py
     (prefetch 4 + persistent for tcga). Requested value is recorded in
     the per-run log.
  4. Mix ordering: wave run-list sorted by predicted VRAM, then
     snake-ordered (heavy/light alternation) so the concurrent window
     mixes big + small jobs.
  5. --max-jobs CLI cap (default 3, max 5). Every admission decision is
     logged to logs/wave_scheduler.log (admitted/deferred + why).
  6. Resume-safe + restartable: runs whose summary JSON already exists
     are skipped; runs launch with --resume (NEVER --no-resume); per-run
     log logs/<label>.log; power-cut recovery = rerun the same command.
  7. --calibrate N: run first N wave jobs with --epochs 2, sample
     VRAM/RAM/epoch-time per concurrent slot, print recommended
     MAX_JOBS + projected wave wall-time.

Mock hooks (for CPU-only verification, no GPU needed):
  WAVE_FAKE_NVIDIA_SMI  path to file "free_mib total_mib"
  WAVE_FAKE_PS          path to file with lines "pid rss_kb"
  WAVE_FAKE_PYTHON      executable used instead of `python` to launch jobs
  WAVE_CYCLE_SECS       scheduler cycle length (default 5)

Usage:
  scripts/run_wave.sh [--dry-run] [--calibrate N] [--max-jobs N] <run-id>...
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
SCHED_LOG = LOGS / "wave_scheduler.log"

# ---------------------------------------------------------------------------
# Measured VRAM table (R3-1 gate, vgg16, MiB) — smoke_test_logs/r31/vram_*.log
# ---------------------------------------------------------------------------
VRAM_TABLE_VGG16 = {"tcga": 8816, "pannuke": 8429, "siim": 6635, "panda": 3205}

# Per-encoder factors derived from the 26-config matrix (run_all_experiments.sh):
# 13 vgg16 configs (measured, factor 1.0) + 13 mobilenet_v2 configs.
# No R3-1 mobilenet measurement exists; 0.45 is the documented derived
# estimate (MobileNetV2 has ~4.4M params vs VGG16's ~138M; activations
# dominate, so the ratio is far above the raw param ratio).
ENCODER_FACTOR = {"vgg16": 1.0, "mobilenet_v2": 0.45}

SAFETY_MARGIN_MIB = 1536          # 1.5 GB VRAM safety margin
GLOBAL_RAM_SOFT_CAP_GB = 17.0     # global soft cap (20 GB RAM - headroom)
TOTAL_PROCS = 8
DEFAULT_MAX_JOBS = 3
MAX_ALLOWED_JOBS = 5
CALIBRATE_EPOCHS = 2
FULL_EPOCHS = 50                  # phase v1/v2 default (src/config.py)

# ---------------------------------------------------------------------------
# Run definitions — byte-for-byte flags from run_all_experiments.sh
# (single-split; --num-workers is replaced by the adaptive value at launch).
# ---------------------------------------------------------------------------
RUN_DEFS: dict[str, tuple[str, str, str, str]] = {
    # id: (name, dataset, encoder, flags-without-num-workers)
    "01": ("g1_tcga_vgg16", "tcga", "vgg16",
           "--phase v1 --datasets tcga --encoders vgg16 --no-macenko --disable-gradnorm --static-weights --compile"),
    "02": ("g1_tcga_mobilenet_v2", "tcga", "mobilenet_v2",
           "--phase v1 --datasets tcga --encoders mobilenet_v2 --no-macenko --disable-gradnorm --static-weights --compile"),
    "03": ("g1_panda_vgg16", "panda", "vgg16",
           "--phase v1 --datasets panda --encoders vgg16 --no-macenko --disable-gradnorm --static-weights --compile"),
    "04": ("g1_panda_mobilenet_v2", "panda", "mobilenet_v2",
           "--phase v1 --datasets panda --encoders mobilenet_v2 --no-macenko --disable-gradnorm --static-weights --compile"),
    "05": ("g1_siim_vgg16", "siim", "vgg16",
           "--phase v1 --datasets siim --encoders vgg16 --no-macenko --disable-gradnorm --static-weights --compile"),
    "06": ("g1_siim_mobilenet_v2", "siim", "mobilenet_v2",
           "--phase v1 --datasets siim --encoders mobilenet_v2 --no-macenko --disable-gradnorm --static-weights --compile"),
    "07": ("g2_tcga_vgg16", "tcga", "vgg16",
           "--phase v2 --datasets tcga --encoders vgg16"),
    "08": ("g2_tcga_mobilenet_v2", "tcga", "mobilenet_v2",
           "--phase v2 --datasets tcga --encoders mobilenet_v2"),
    "09": ("g2_panda_vgg16", "panda", "vgg16",
           "--phase v2 --datasets panda --encoders vgg16"),
    "10": ("g2_panda_mobilenet_v2", "panda", "mobilenet_v2",
           "--phase v2 --datasets panda --encoders mobilenet_v2"),
    "11": ("g2_siim_vgg16", "siim", "vgg16",
           "--phase v2 --datasets siim --encoders vgg16"),
    "12": ("g2_siim_mobilenet_v2", "siim", "mobilenet_v2",
           "--phase v2 --datasets siim --encoders mobilenet_v2"),
    "13": ("g3_pannuke_vgg16_naked", "pannuke", "vgg16",
           "--phase v1 --datasets pannuke --encoders vgg16 --no-macenko --disable-gradnorm --static-weights --compile"),
    "14": ("g3_pannuke_mobilenet_v2_naked", "pannuke", "mobilenet_v2",
           "--phase v1 --datasets pannuke --encoders mobilenet_v2 --no-macenko --disable-gradnorm --static-weights --compile"),
    "15": ("g3_pannuke_vgg16_final", "pannuke", "vgg16",
           "--phase v2 --datasets pannuke --encoders vgg16"),
    "16": ("g3_pannuke_mobilenet_v2_final", "pannuke", "mobilenet_v2",
           "--phase v2 --datasets pannuke --encoders mobilenet_v2"),
    "17": ("g4_panda_isolate_lr", "panda", "vgg16",
           "--phase v2 --datasets panda --encoders vgg16 --no-macenko --disable-gradnorm --static-weights"),
    "18": ("g4_panda_isolate_gn", "panda", "vgg16",
           "--phase v1 --datasets panda --encoders vgg16 --no-macenko --enable-gradnorm"),
    "19": ("g4_panda_lambda_1_1", "panda", "vgg16",
           "--phase v1 --datasets panda --encoders vgg16 --no-macenko --disable-gradnorm --static-weights --lambda-seg 1 --lambda-cls 1 --compile"),
    "20": ("g4_panda_lambda_5_1", "panda", "vgg16",
           "--phase v1 --datasets panda --encoders vgg16 --no-macenko --disable-gradnorm --static-weights --lambda-seg 5 --lambda-cls 1 --compile"),
    "21": ("g4_panda_lambda_1_10", "panda", "vgg16",
           "--phase v1 --datasets panda --encoders vgg16 --no-macenko --disable-gradnorm --static-weights --lambda-seg 1 --lambda-cls 10 --compile"),
    "22": ("g4_panda_lambda_10_1", "panda", "vgg16",
           "--phase v1 --datasets panda --encoders vgg16 --no-macenko --disable-gradnorm --static-weights --lambda-seg 10 --lambda-cls 1 --compile"),
    "23": ("g5_panda_nomacenko", "panda", "mobilenet_v2",
           "--phase v2 --datasets panda --encoders mobilenet_v2 --no-macenko"),
    "24": ("g5_pannuke_nomacenko", "pannuke", "mobilenet_v2",
           "--phase v2 --datasets pannuke --encoders mobilenet_v2 --no-macenko"),
    "25": ("g5_tcga_noskip", "tcga", "mobilenet_v2",
           "--phase v2 --datasets tcga --encoders mobilenet_v2 --no-skip-connections"),
    "26": ("g5_panda_noskip", "panda", "mobilenet_v2",
           "--phase v2 --datasets panda --encoders mobilenet_v2 --no-skip-connections"),
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Job:
    run_id: str
    name: str
    dataset: str
    encoder: str
    flags: str
    seq: int                      # original request order (tie-break)
    predicted_vram: int = 0       # MiB
    label: str = ""
    log_file: str = ""
    summary_file: str = ""
    pid: int | None = None
    proc: "subprocess.Popen | None" = None
    workers: int = 0
    state: str = "pending"        # pending|active|done|skipped|aborted
    started: float = 0.0
    wall_sec: float = 0.0
    peak_rss_kb: int = 0
    extra: dict = field(default_factory=dict)

    def summary_exists(self) -> bool:
        return (ROOT / self.summary_file).exists()


def predict_vram(dataset: str, encoder: str) -> int:
    base = VRAM_TABLE_VGG16.get(dataset)
    if base is None:
        # Unknown dataset: conservative fallback = largest measured.
        base = max(VRAM_TABLE_VGG16.values())
    factor = ENCODER_FACTOR.get(encoder, 1.0)
    return int(round(base * factor))


def workers_for(active_jobs: int) -> int:
    """Adaptive dataloader workers: 8 procs / active jobs (persistent, prefetch 4)."""
    if active_jobs <= 1:
        return 6
    if active_jobs == 2:
        return 3
    if active_jobs <= 4:
        return 2
    return 1


def snake_order(jobs: list[Job]) -> list[Job]:
    """Sort by predicted VRAM desc (ties by request order), then interleave
    heavy/light from both ends so the concurrent window mixes big + small."""
    s = sorted(jobs, key=lambda j: (-j.predicted_vram, j.seq))
    out: list[Job] = []
    i, j, take_heavy = 0, len(s) - 1, True
    while i <= j:
        if take_heavy:
            out.append(s[i])
            i += 1
        else:
            out.append(s[j])
            j -= 1
        take_heavy = not take_heavy
    return out


# ---------------------------------------------------------------------------
# Telemetry (mockable)
# ---------------------------------------------------------------------------
def query_vram() -> tuple[int, int]:
    """Return (free_mib, total_mib). WAVE_FAKE_NVIDIA_SMI fixture overrides."""
    fake = os.environ.get("WAVE_FAKE_NVIDIA_SMI")
    if fake:
        try:
            free, total = Path(fake).read_text().split()
            return int(float(free)), int(float(total))
        except Exception:
            pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        free, total = out.stdout.strip().split(",")
        return int(float(free)), int(float(total))
    except Exception:
        return 24576, 24576  # no GPU visible: assume full 24 GB free


def sample_rss_kb(pid: int) -> int:
    """RSS in KB for a pid. WAVE_FAKE_PS fixture (lines 'pid rss_kb') overrides."""
    fake = os.environ.get("WAVE_FAKE_PS")
    if fake:
        try:
            for line in Path(fake).read_text().splitlines():
                parts = line.split()
                if len(parts) >= 2 and int(parts[0]) == pid:
                    return int(float(parts[1]))
        except Exception:
            pass
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip() or 0)
    except Exception:
        return 0


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def job_alive(job: "Job") -> bool:
    """True while the job's process is still running.

    Uses Popen.poll() (not os.kill(pid,0)) because the job is a CHILD of the
    scheduler: an exited child becomes a ZOMBIE until reaped, and
    os.kill(pid,0) still succeeds for a zombie. poll() reaps it correctly.
    """
    if job.proc is not None:
        return job.proc.poll() is None
    if job.pid is not None:
        return pid_alive(job.pid)
    return False


def log(msg: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(SCHED_LOG, "a") as f:
        f.write(line + "\n")
    print(line)


# ---------------------------------------------------------------------------
# Launch / kill
# ---------------------------------------------------------------------------
def build_cmd(job: Job, calibrate: bool = False) -> list[str]:
    py = os.environ.get("WAVE_FAKE_PYTHON") or "python"
    cmd = [py, "main.py"] + job.flags.split()
    cmd += ["--num-workers", str(job.workers)]
    if calibrate:
        cmd += ["--epochs", str(CALIBRATE_EPOCHS)]
    # Resume-safe: --resume by default (NEVER --no-resume).
    cmd += ["--resume",
            "--summary-out", job.summary_file,
            "--run-label", job.label]
    return cmd


def launch(job: Job, calibrate: bool = False) -> None:
    cmd = build_cmd(job, calibrate)
    (ROOT / job.log_file).parent.mkdir(parents=True, exist_ok=True)
    log_file = open(ROOT / job.log_file, "a")
    log_file.write(f"[{job.run_id}] {time.strftime('%Y-%m-%d %H:%M:%S')} - "
                   f"START (wave, resume-safe) workers={job.workers} "
                   f"predicted_vram={job.predicted_vram}MiB\n")
    log_file.write(f"[{job.run_id}] Command: {' '.join(cmd)}\n")
    log_file.flush()
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log_file,
                            stderr=subprocess.STDOUT)
    log_file.close()
    job.pid = proc.pid
    job.proc = proc
    job.state = "active"
    job.started = time.time()
    log(f"LAUNCH {job.run_id} {job.name} pid={proc.pid} "
        f"workers={job.workers} predicted={job.predicted_vram}MiB")


def kill_job(job: Job, reason: str) -> None:
    if job.pid is None:
        return
    try:
        os.kill(job.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    # Reap the killed child so it does not linger as a zombie.
    if job.proc is not None:
        try:
            job.proc.wait(timeout=5)
        except Exception:
            pass
    job.state = "aborted"
    job.wall_sec = time.time() - job.started
    log(f"ABORT {job.run_id} {job.name} pid={job.pid} reason={reason}")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def resolve_jobs(run_ids: list[str]) -> list[Job]:
    jobs = []
    for seq, rid in enumerate(run_ids):
        rid = f"{int(rid):02d}"
        if rid not in RUN_DEFS:
            raise SystemExit(f"ERROR: unknown run-id '{rid}' (expected 01..26)")
        name, dataset, encoder, flags = RUN_DEFS[rid]
        # WAVE_OUT_ROOT: test/mock hook to redirect per-run outputs.
        out_root = os.environ.get("WAVE_OUT_ROOT")
        log_dir = out_root if out_root else "logs"
        sum_dir = out_root if out_root else "checkpoints"
        j = Job(run_id=rid, name=name, dataset=dataset, encoder=encoder,
                flags=flags, seq=seq,
                predicted_vram=predict_vram(dataset, encoder),
                label=f"{rid}_{name}",
                log_file=f"{log_dir}/{rid}_{name}.log",
                summary_file=f"{sum_dir}/summary_{rid}_{name}.json")
        jobs.append(j)
    return jobs


def print_dry_run(jobs: list[Job], max_jobs: int) -> None:
    free, total = query_vram()
    budget = free - SAFETY_MARGIN_MIB
    print("=" * 72)
    print(f" [DRY-RUN] Wave admission plan  (free={free} MiB / total={total} MiB, "
          f"safety margin={SAFETY_MARGIN_MIB} MiB, max_jobs={max_jobs})")
    print(" " + "-" * 70)
    print(f"  {'ord':>3} {'id':>3} {'name':<28} {'pred MiB':>9} {'workers':>7}  decision")
    print(" " + "-" * 70)
    ordered = snake_order(jobs)
    admitted = 0
    cum = 0
    for n, j in enumerate(ordered, 1):
        if j.summary_exists():
            decision = "SKIP (summary exists)"
        elif admitted < max_jobs and cum + j.predicted_vram <= budget:
            decision = "ADMIT"
            admitted += 1
            cum += j.predicted_vram
        else:
            decision = "DEFER (re-checked each cycle)"
        w = workers_for(min(max(admitted, 1), max_jobs))
        print(f"  {n:>3} {j.run_id:>3} {j.name:<28} {j.predicted_vram:>9} "
              f"{w:>7}  {decision}")
    print(" " + "-" * 70)
    print(f" [DRY-RUN] {len(ordered)} run(s) planned; "
          f"{admitted} admitted in first window, "
          f"{len(ordered) - admitted} deferred/skipped. Nothing launched.")
    print("=" * 72)


def run_calibrate(jobs: list[Job], max_jobs: int, n: int) -> None:
    """Run first N wave jobs with --epochs 2; sample VRAM/RAM/epoch-time per
    concurrent slot; print recommended MAX_JOBS + projected wave wall-time."""
    n = min(n, max_jobs, len(jobs))
    cal_jobs = snake_order(jobs)[:n]
    for j in cal_jobs:
        j.workers = workers_for(n)
    free0, total = query_vram()
    log(f"CALIBRATE start n={n} free={free0}MiB total={total}MiB")
    for j in cal_jobs:
        launch(j, calibrate=True)
    min_free = free0
    sample_every = 0.5  # fine-grained sampling so short RSS/VRAM spikes are caught
    t0 = time.time()
    while any(j.state == "active" for j in cal_jobs):
        for j in cal_jobs:
            if j.state == "active":
                if not job_alive(j):
                    j.state = "done"
                    j.wall_sec = time.time() - j.started
                else:
                    j.peak_rss_kb = max(j.peak_rss_kb, sample_rss_kb(j.pid))
        f, _ = query_vram()
        min_free = min(min_free, f)
        time.sleep(sample_every)
    elapsed = time.time() - t0
    vram_used = total - min_free
    avg_vram = vram_used / n
    avg_rss_gb = sum(j.peak_rss_kb for j in cal_jobs) / n / (1024 * 1024)
    avg_epoch_sec = sum(j.wall_sec for j in cal_jobs) / n / CALIBRATE_EPOCHS
    # Recommended concurrency: VRAM-bound and RAM-bound, capped at 5.
    rec_vram = int((total - SAFETY_MARGIN_MIB) // max(1, avg_vram))
    rec_ram = int(GLOBAL_RAM_SOFT_CAP_GB / max(0.001, avg_rss_gb))
    rec = max(1, min(MAX_ALLOWED_JOBS, rec_vram, rec_ram))
    # Projected full-wave wall-time: sum(per-job full time) / rec concurrency.
    full_total = 0.0
    for j in jobs:
        if j.summary_exists():
            continue
        est = avg_epoch_sec * FULL_EPOCHS
        full_total += est
    projected_h = full_total / max(1, rec) / 3600.0
    report = {
        "n": n, "vram_used_mib": vram_used, "avg_vram_mib": round(avg_vram, 1),
        "avg_rss_gb": round(avg_rss_gb, 2),
        "avg_epoch_sec": round(avg_epoch_sec, 1),
        "recommended_max_jobs": rec,
        "projected_wave_wall_h": round(projected_h, 2),
        "per_job": [
            {"id": j.run_id, "wall_sec": round(j.wall_sec, 1),
             "peak_rss_gb": round(j.peak_rss_kb / 1024 / 1024, 2)}
            for j in cal_jobs],
    }
    out = LOGS / "wave_calibrate.json"
    out.write_text(json.dumps(report, indent=2))
    log(f"CALIBRATE done: {json.dumps(report)}")
    print("=" * 72)
    print(f" [CALIBRATE] n={n}  avg VRAM/job={avg_vram:.0f} MiB  "
          f"avg RSS/job={avg_rss_gb:.2f} GB  avg epoch={avg_epoch_sec:.1f}s")
    print(f" [CALIBRATE] Recommended MAX_JOBS: {rec}  "
          f"(VRAM-bound={rec_vram}, RAM-bound={rec_ram}, cap={MAX_ALLOWED_JOBS})")
    print(f" [CALIBRATE] Projected wave wall-time: {projected_h:.2f} h")
    print(f" [CALIBRATE] Report: {out}")
    print("=" * 72)


def run_wave(jobs: list[Job], max_jobs: int) -> int:
    cycle = float(os.environ.get("WAVE_CYCLE_SECS", "5"))
    budget_kb = int(GLOBAL_RAM_SOFT_CAP_GB * 1024 * 1024 / max(1, max_jobs))
    pending = snake_order(jobs)
    active: list[Job] = []
    failures = 0
    log(f"WAVE start: {len(pending)} pending, max_jobs={max_jobs}, "
        f"per-job RAM budget={budget_kb / 1024 / 1024:.2f} GB, "
        f"global soft cap={GLOBAL_RAM_SOFT_CAP_GB} GB")
    while pending or active:
        # 1) reap finished (poll() reaps the child; os.kill would see zombies)
        for j in list(active):
            if not job_alive(j):
                j.state = "done"
                j.wall_sec = time.time() - j.started
                active.remove(j)
                log(f"DONE {j.run_id} {j.name} wall={j.wall_sec:.0f}s "
                    f"peak_rss={j.peak_rss_kb / 1024 / 1024:.2f}GB")
        # 2) RAM watchdog
        total_rss_kb = 0
        for j in active:
            rss = sample_rss_kb(j.pid) if j.pid is not None else 0
            j.peak_rss_kb = max(j.peak_rss_kb, rss)
            total_rss_kb += rss
            if rss > budget_kb:
                kill_job(j, f"RAM budget breach: {rss / 1024 / 1024:.2f} GB "
                            f"> {budget_kb / 1024 / 1024:.2f} GB (aborted alone; wave continues)")
                active.remove(j)
                failures += 1
        if total_rss_kb > GLOBAL_RAM_SOFT_CAP_GB * 1024 * 1024:
            log(f"WARNING global RAM soft cap breached: "
                f"{total_rss_kb / 1024 / 1024:.2f} GB > {GLOBAL_RAM_SOFT_CAP_GB} GB")
        # 3) VRAM-aware admission (defer, never abort)
        if len(active) < max_jobs:
            for j in list(pending):
                if len(active) >= max_jobs:
                    break
                free, _ = query_vram()
                if free - SAFETY_MARGIN_MIB >= j.predicted_vram:
                    j.workers = workers_for(len(active) + 1)
                    pending.remove(j)
                    active.append(j)
                    launch(j)
                    log(f"ADMITTED {j.run_id} {j.name}: free={free}MiB "
                        f"margin={SAFETY_MARGIN_MIB}MiB "
                        f"predicted={j.predicted_vram}MiB "
                        f"workers={j.workers} (fits: {free - SAFETY_MARGIN_MIB} >= {j.predicted_vram})")
                else:
                    log(f"DEFERRED {j.run_id} {j.name}: free={free}MiB - "
                        f"margin={SAFETY_MARGIN_MIB}MiB = {free - SAFETY_MARGIN_MIB}MiB "
                        f"< predicted={j.predicted_vram}MiB (re-checked next cycle)")
                    # Per-job defer: keep trying lighter pending jobs (snake
                    # order) so a non-fitting heavy job never stalls the wave.
        if pending or active:
            time.sleep(cycle)
    log(f"WAVE complete: {len(jobs) - failures} ok, {failures} aborted")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    dry_run = False
    calibrate: int | None = None
    max_jobs = DEFAULT_MAX_JOBS
    run_ids: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--dry-run":
            dry_run = True
        elif a == "--calibrate":
            i += 1
            calibrate = int(argv[i])
        elif a == "--max-jobs":
            i += 1
            max_jobs = int(argv[i])
        else:
            run_ids.append(a)
        i += 1
    if not run_ids:
        print("Usage: run_wave.sh [--dry-run] [--calibrate N] [--max-jobs N] <run-id>...")
        return 2
    if not 1 <= max_jobs <= MAX_ALLOWED_JOBS:
        print(f"ERROR: --max-jobs must be 1..{MAX_ALLOWED_JOBS} (got {max_jobs})")
        return 2
    jobs = resolve_jobs(run_ids)
    # Resume-safe: skip runs whose summary JSON already exists.
    for j in jobs:
        if j.summary_exists():
            j.state = "skipped"
            log(f"SKIP {j.run_id} {j.name}: summary exists ({j.summary_file})")
    remaining = [j for j in jobs if j.state != "skipped"]
    if dry_run:
        print_dry_run(jobs, max_jobs)
        return 0
    if calibrate is not None:
        if not remaining:
            print(" [CALIBRATE] nothing to calibrate (all summaries exist)")
            return 0
        run_calibrate(remaining, max_jobs, calibrate)
        return 0
    if not remaining:
        print(" [WAVE] all requested runs already complete (summaries exist)")
        return 0
    return run_wave(remaining, max_jobs)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
