"""Convenience script to run the streaming CTC spike over a directory of wav files
and log aggregated metrics.

Usage:
python -m w2v2-ctc_workflow.run_spike_on_directory --dir dist/organized_recordings [batch-options] [spike-extra-args]

Examples:
# Forward extra args (chunk/stride) directly without --extra separator
python -m w2v2-ctc_workflow.run_spike_on_directory \
  --dir dist/organized_recordings --ignore-aug --recursive \
  --log-dir logs/spike_runs --chunk_ms 30 --stride_ms 15

Optional batch options:
--pattern "*.wav" (default)
--recursive (to traverse subdirectories)
--limit N (process only first N files)
--sleep 0.0 (delay between files)
--log-dir logs/spike_runs (store per-file + summary logs)
--ignore-aug (skip files whose name starts with aug_)

Any unknown flags are forwarded to the spike module.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime


def run_file(py_exe: str, spike_module: str, wav_path: Path, extra_args: list[str]) -> tuple[int, float, str]:
    """Run the spike script for a single wav file.

    Returns:
        (exit_code, elapsed_seconds, combined_output)
    """
    cmd = [py_exe, "-m", spike_module, "--wav", str(wav_path), *extra_args]
    start = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - start
    output = proc.stdout
    print(f"===== OUTPUT for {wav_path.name} =====")
    print(output)
    return proc.returncode, elapsed, output


def collect_wavs(root: Path, pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        return sorted([p for p in root.rglob(pattern) if p.is_file()])
    return sorted([p for p in root.glob(pattern) if p.is_file()])


def parse_args():
    ap = argparse.ArgumentParser(description="Batch runner for streaming CTC spike", add_help=True)
    ap.add_argument("--dir", required=True, help="Directory containing wav files")
    ap.add_argument("--pattern", default="*.wav", help="Glob pattern (default: *.wav)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    ap.add_argument("--limit", type=int, default=0, help="Max number of files (0 = all)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between files")
    ap.add_argument("--spike-module", default="w2v2-ctc_workflow.stream_wav2vec2_ctc_spike", help="Module path of spike script")
    ap.add_argument("--log-dir", type=str, default="", help="If set, write per-file logs + summary here")
    ap.add_argument("--ignore-aug", action="store_true", help="Ignore wav files starting with aug_")
    args, extra = ap.parse_known_args()
    setattr(args, "extra", extra)
    if extra:
        print(f"Forwarding extra spike args: {' '.join(extra)}")
    return args


def init_log_dir(log_dir: str) -> Path | None:
    if not log_dir:
        return None
    p = Path(log_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_file(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    root = Path(args.dir)
    if not root.exists():
        print(f"Directory not found: {root}")
        sys.exit(1)
    wavs = collect_wavs(root, args.pattern, args.recursive)
    if args.ignore_aug:
        wavs = [w for w in wavs if not w.name.startswith("aug_")]
    if args.limit > 0:
        wavs = wavs[:args.limit]
    if not wavs:
        print("No wav files found.")
        sys.exit(1)

    log_dir = init_log_dir(args.log_dir)
    if log_dir:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_dir = log_dir / f"run_{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Logging to {log_dir}")

    print(f"Found {len(wavs)} wav files. Starting processing...\n")

    py_exe = sys.executable
    spike_module = args.spike_module
    extra_args = getattr(args, 'extra', [])

    all_times: list[float] = []
    failures = 0
    per_file_records: list[str] = []

    batch_start = time.time()
    for i, wav in enumerate(wavs, 1):
        print(f"[{i}/{len(wavs)}] {wav}")
        code, elapsed, output = run_file(py_exe, spike_module, wav, extra_args)
        rec = f"{i}\t{wav}\t{elapsed:.3f}\t{code}"
        per_file_records.append(rec)
        if code != 0:
            failures += 1
            print(f"ERROR: spike failed on {wav} (exit {code})")
        else:
            all_times.append(elapsed)
        if log_dir:
            file_log = log_dir / f"{i:04d}_{wav.stem}.log"
            write_file(file_log, output)
        if args.sleep > 0:
            time.sleep(args.sleep)

    total_elapsed = time.time() - batch_start
    print("\nBatch complete.")
    print(f"Success: {len(all_times)}  Failures: {failures}")
    if all_times:
        print(f"Per-file elapsed (s): min={min(all_times):.2f} max={max(all_times):.2f} mean={statistics.mean(all_times):.2f}")
        if len(all_times) > 1:
            print(f"Median={statistics.median(all_times):.2f} stdev={statistics.pstdev(all_times):.2f}")
    print(f"Total batch wall time: {total_elapsed:.2f}s")

    if log_dir:
        summary_lines = [
            "# Batch Summary",
            f"Total files: {len(wavs)}",
            f"Success: {len(all_times)}",
            f"Failures: {failures}",
            f"Total wall time: {total_elapsed:.2f}s",
        ]
        if all_times:
            summary_lines.append(
                f"Perf: min={min(all_times):.3f}s max={max(all_times):.3f}s mean={statistics.mean(all_times):.3f}s"
            )
            if len(all_times) > 1:
                summary_lines.append(
                    f"median={statistics.median(all_times):.3f}s stdev={statistics.pstdev(all_times):.3f}s"
                )
        summary_lines.append("\n# Per-file (index\tpath\telapsed_s\texit_code)")
        summary_lines.extend(per_file_records)
        write_file(log_dir / "summary.txt", "\n".join(summary_lines))
        print(f"Wrote summary: {log_dir / 'summary.txt'}")


if __name__ == "__main__":  # pragma: no cover
    main()
