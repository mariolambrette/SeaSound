#!/usr/bin/env python3
"""
profile_run.py — run a command-line tool unchanged and profile its entire
process tree (the launcher/coordinator and every worker it spawns).

The command under test is passed verbatim after `--`, so it matches real
deployment exactly:

    python profile_run.py -- seasound --config path\\to\\config.yaml --auto

Optional flags (put BEFORE the `--`):
    --interval 1.0              sampling period in seconds
    --outdir profile_results    where the CSVs + plot are written
    --label my_test_run         name used in output filenames
    --watts-per-core 6.0        if set, adds a second rough energy estimate

Outputs (all CSV, plus an optional PNG):
    <label>_timeseries.csv      one row per sample (whole-tree aggregate)
    <label>_per_process.csv     peak memory for each process seen
    <label>_summary.csv         one-row headline metrics for collaborators
    <label>_profile.png         memory / cores / process-count over time

Requires: psutil       (pip install psutil)
Optional: matplotlib   (for the plot)
          codecarbon   (for the kWh estimate — pip install codecarbon)
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime

import psutil


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=float, default=1.0,
                    help="sampling period in seconds (default 1.0)")
    ap.add_argument("--outdir", default="profile_results")
    ap.add_argument("--label", default=None)
    ap.add_argument("--watts-per-core", type=float, default=None,
                    help="assumed watts per busy core for a second energy estimate")
    ap.add_argument("--detailed", action="store_true",
                    help="log per-process memory/CPU every tick and plot the heaviest")
    ap.add_argument("--top-n", type=int, default=8,
                    help="how many top memory processes to plot in detailed mode")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="the command to run, preceded by --")
    args = ap.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        sys.exit("No command given.\n"
                 "Example: python profile_run.py -- seasound --config cfg.yaml --auto")
    return args, cmd


def main():
    args, cmd = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    label = args.label or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    ts_path = os.path.join(args.outdir, f"{label}_timeseries.csv")
    proc_path = os.path.join(args.outdir, f"{label}_per_process.csv")
    summary_path = os.path.join(args.outdir, f"{label}_summary.csv")
    detail_path = os.path.join(args.outdir, f"{label}_per_process_timeseries.csv")

    n_logical = psutil.cpu_count(logical=True)
    total_ram_gb = psutil.virtual_memory().total / 1e9

    # ---- optional CodeCarbon tracker (machine-level energy over the run) ----
    tracker = None
    try:
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker(measure_power_secs=int(max(1, args.interval)),
                                   save_to_file=False, log_level="error")
        tracker.start()
    except ImportError:
        pass
    except Exception as e:
        print(f"CodeCarbon could not start ({e}); continuing without it.")
        tracker = None

    print(f"Machine: {n_logical} logical cores, {total_ram_gb:.1f} GB RAM")
    print(f"Launching: {' '.join(cmd)}\n")

    t0 = time.time()
    try:
        proc = subprocess.Popen(cmd)
    except FileNotFoundError:
        sys.exit(f"Could not find '{cmd[0]}'. If it is a .bat/.cmd, run it as:\n"
                 f"  python profile_run.py -- cmd /c {cmd[0]} ...")

    root_pid = proc.pid
    root = psutil.Process(root_pid)
    tracked = {}                 # pid -> persistent psutil.Process (needed for cpu_percent)
    samples = []                 # per-tick aggregate
    proc_info = {}               # pid -> {name, role, cmdline, first_seen, peak_rss}
    pid_series = {}              # pid -> list of (t, rss_mb)  (detailed mode only)

    def short_cmdline(p):
        try:
            parts = p.cmdline()
            s = " ".join(parts) if parts else p.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            s = ""
        return s[:300]

    detail_f = detail_writer = None
    if args.detailed:
        detail_f = open(detail_path, "w", newline="")
        detail_writer = csv.writer(detail_f)
        detail_writer.writerow(["elapsed_s", "pid", "role", "name",
                                "rss_mb", "cpu_pct"])

    with open(ts_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["elapsed_s", "n_procs", "total_rss_mb",
                         "total_cpu_pct", "cores_busy"])
        try:
            while proc.poll() is None:
                tick = time.time() - t0
                try:
                    current = [root] + root.children(recursive=True)
                except psutil.NoSuchProcess:
                    break

                current_pids = set()
                total_rss = total_cpu = 0.0
                alive = 0
                for p in current:
                    pid = p.pid
                    current_pids.add(pid)
                    # Reuse a persistent Process object per PID. cpu_percent()
                    # compares against the previous call ON THE SAME OBJECT, so
                    # the object must survive between ticks or it always reads 0.
                    is_new = pid not in tracked
                    if is_new:
                        tracked[pid] = p
                    proc_obj = tracked[pid]
                    try:
                        if is_new:
                            proc_obj.cpu_percent()       # prime; first call is 0.0
                            cpu = 0.0
                            proc_info[pid] = {
                                "name": proc_obj.name(),
                                "role": "coordinator" if pid == root_pid else "worker",
                                "cmdline": short_cmdline(proc_obj),
                                "first_seen_s": round(tick, 1),
                                "peak_rss_mb": 0.0,
                            }
                        else:
                            cpu = proc_obj.cpu_percent()  # since previous tick
                        rss = proc_obj.memory_info().rss / 1e6
                        total_rss += rss
                        total_cpu += cpu
                        alive += 1
                        info = proc_info.get(pid)
                        if info:
                            info["peak_rss_mb"] = max(info["peak_rss_mb"], rss)
                        if args.detailed:
                            detail_writer.writerow([round(tick, 1), pid,
                                                    info["role"], info["name"],
                                                    round(rss, 1), round(cpu, 1)])
                            pid_series.setdefault(pid, []).append((tick, rss))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                # drop processes that have exited so the cache doesn't grow
                for dead in set(tracked) - current_pids:
                    tracked.pop(dead, None)

                cores_busy = total_cpu / 100.0
                writer.writerow([round(tick, 1), alive, round(total_rss, 1),
                                 round(total_cpu, 1), round(cores_busy, 2)])
                f.flush()
                samples.append({"t": tick, "n": alive, "rss": total_rss,
                                "cores": cores_busy})
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nInterrupted — terminating the command under test...")
            proc.terminate()

    if detail_f is not None:
        detail_f.close()

    wall = time.time() - t0
    rc = proc.returncode

    # ---- stop CodeCarbon and read kWh ----
    energy_kwh = None
    if tracker is not None:
        try:
            tracker.stop()
            data = getattr(tracker, "final_emissions_data", None)
            if data is not None:
                energy_kwh = getattr(data, "energy_consumed", None)
        except Exception as e:
            print(f"CodeCarbon error on stop: {e}")

    if not samples:
        sys.exit("The command finished before the first sample. Lower --interval.")

    # ---- memory metrics ----
    peak_total_gb = max(s["rss"] for s in samples) / 1000.0      # parallel requirement
    coordinator_peak = proc_info.get(root_pid, {}).get("peak_rss_mb", 0.0)
    worker_peaks = [v["peak_rss_mb"] for k, v in proc_info.items()
                    if v["role"] == "worker"]
    max_worker_peak = max(worker_peaks) if worker_peaks else 0.0
    # sequential-run floor: coordinator stays resident + one worker at a time
    seq_floor_gb = (coordinator_peak + max_worker_peak) / 1000.0

    # ---- compute metrics ----
    peak_cores = max(s["cores"] for s in samples)
    mean_cores = sum(s["cores"] for s in samples) / len(samples)
    core_hours = sum(s["cores"] for s in samples) * args.interval / 3600.0

    # ---- write per-process CSV ----
    with open(proc_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pid", "name", "role", "first_seen_s", "peak_rss_mb", "cmdline"])
        for pid, v in sorted(proc_info.items(),
                             key=lambda kv: kv[1]["peak_rss_mb"], reverse=True):
            w.writerow([pid, v["name"], v["role"], v["first_seen_s"],
                        round(v["peak_rss_mb"], 1), v.get("cmdline", "")])

    # ---- write summary CSV ----
    summary = {
        "label": label,
        "exit_code": rc,
        "wall_clock_s": round(wall, 1),
        "n_logical_cores": n_logical,
        "machine_ram_gb": round(total_ram_gb, 1),
        "peak_total_mem_gb": round(peak_total_gb, 3),
        "coordinator_peak_mem_mb": round(coordinator_peak, 1),
        "max_worker_peak_mem_mb": round(max_worker_peak, 1),
        "sequential_floor_mem_gb": round(seq_floor_gb, 3),
        "peak_cores_busy": round(peak_cores, 2),
        "mean_cores_busy": round(mean_cores, 2),
        "core_hours": round(core_hours, 4),
        "codecarbon_kwh_ESTIMATE": ("" if energy_kwh is None else round(energy_kwh, 5)),
        "watts_per_core_kwh_ESTIMATE": (
            round(core_hours * args.watts_per_core / 1000.0, 5)
            if args.watts_per_core else ""),
    }
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    # ---- console summary ----
    print("\n" + "=" * 56)
    print(f"  SUMMARY — {label}")
    print("=" * 56)
    print(f"  Exit code                 : {rc}")
    print(f"  Wall-clock time           : {wall:.1f} s  ({wall/60:.1f} min)")
    print("  -- memory --")
    print(f"  Peak total (all procs)    : {peak_total_gb:.2f} GB   <- parallel RAM need")
    print(f"  Coordinator peak          : {coordinator_peak/1000:.2f} GB")
    print(f"  Heaviest single worker    : {max_worker_peak/1000:.2f} GB")
    print(f"  Sequential-run floor      : {seq_floor_gb:.2f} GB   <- if run one job at a time")
    print("  -- compute --")
    print(f"  Peak cores in use         : {peak_cores:.1f} of {n_logical}")
    print(f"  Mean cores in use         : {mean_cores:.1f} of {n_logical}")
    print(f"  Total compute             : {core_hours:.3f} core-hours")
    print("  -- energy (ESTIMATES, treat as rough) --")
    if energy_kwh is not None:
        print(f"  CodeCarbon                : {energy_kwh:.4f} kWh  (machine-level, see notes)")
    else:
        print(f"  CodeCarbon                : not run (pip install codecarbon)")
    if args.watts_per_core:
        print(f"  watts-per-core estimate   : {core_hours*args.watts_per_core/1000:.4f} kWh"
              f"  (@ {args.watts_per_core} W/busy-core)")
    print("=" * 56)
    if args.detailed:
        print("  Top processes by peak memory:")
        ranked = sorted(proc_info.items(),
                        key=lambda kv: kv[1]["peak_rss_mb"], reverse=True)
        for pid, v in ranked[:args.top_n]:
            cmd_disp = v.get("cmdline", "") or v["name"]
            print(f"    {v['peak_rss_mb']/1000:6.2f} GB  pid {pid:<7} {v['role']:<11} "
                  f"{cmd_disp[:70]}")
        print("=" * 56)
    print(f"  {ts_path}")
    print(f"  {proc_path}")
    print(f"  {summary_path}")
    if args.detailed:
        print(f"  {detail_path}")

    # ---- optional plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = [s["t"] for s in samples]
        fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        ax[0].plot(t, [s["rss"] / 1000.0 for s in samples], color="#2563eb")
        ax[0].axhline(seq_floor_gb, ls=":", lw=0.9, color="#2563eb",
                      label=f"sequential floor {seq_floor_gb:.1f} GB")
        ax[0].set_ylabel("Memory (GB)")
        ax[0].set_title(f"Compute profile — {label}")
        ax[0].legend(loc="upper right", fontsize=8)
        ax[1].plot(t, [s["cores"] for s in samples], color="#dc2626")
        ax[1].axhline(n_logical, ls="--", lw=0.8, color="grey",
                      label=f"{n_logical} cores available")
        ax[1].set_ylabel("Cores busy")
        ax[1].legend(loc="upper right", fontsize=8)
        ax[2].plot(t, [s["n"] for s in samples], color="#059669")
        ax[2].set_ylabel("Processes")
        ax[2].set_xlabel("Elapsed time (s)")
        for a in ax:
            a.grid(alpha=0.3)
        fig.tight_layout()
        png_path = os.path.join(args.outdir, f"{label}_profile.png")
        fig.savefig(png_path, dpi=130)
        print(f"  {png_path}")

        # detailed: memory-over-time for the heaviest processes
        if args.detailed and pid_series:
            top = sorted(pid_series,
                         key=lambda pid: proc_info.get(pid, {}).get("peak_rss_mb", 0),
                         reverse=True)[:args.top_n]
            fig2, ax2 = plt.subplots(figsize=(9, 5))
            for pid in top:
                series = pid_series[pid]
                xs = [pt[0] for pt in series]
                ys = [pt[1] / 1000.0 for pt in series]
                info = proc_info.get(pid, {})
                lbl = f"pid {pid} ({info.get('role','?')})"
                ax2.plot(xs, ys, lw=1.2, label=lbl)
            ax2.set_xlabel("Elapsed time (s)")
            ax2.set_ylabel("Memory (GB)")
            ax2.set_title(f"Per-process memory — top {len(top)} — {label}")
            ax2.grid(alpha=0.3)
            ax2.legend(loc="upper right", fontsize=7, ncol=2)
            fig2.tight_layout()
            pp_path = os.path.join(args.outdir, f"{label}_per_process_profile.png")
            fig2.savefig(pp_path, dpi=130)
            print(f"  {pp_path}")
    except ImportError:
        print("  (install matplotlib for the plot)")

    sys.exit(rc)


if __name__ == "__main__":
    main()
