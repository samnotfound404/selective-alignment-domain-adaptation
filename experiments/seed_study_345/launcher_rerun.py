#!/usr/bin/env python3
"""Runs all (method, N, seed) jobs of the five-seed study, packing as many
jobs onto each of the 4 GPUs as free memory allows. Resumable: a job whose
result file exists is skipped. A failed job is retried once."""
import os
import subprocess
import sys
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PY = sys.executable
ROOT = os.environ.get("CCSATDA_STUDY_DIR", os.path.join(REPO, "results", "seed_study_345"))
SCRIPT = os.path.join(HERE, "run_study.py")
SEEDS = [42, 123, 456, 789, 1011]
NS = ["full", "200", "100", "50"]          # longest first
METHODS = ["cc", "satda", "sup"]
GPUS = [0, 1, 2, 3]
MIN_FREE_MB = 6500        # launch only if this much is free on the GPU
MAX_PER_GPU = 2
JOB_ENV = dict(os.environ, OMP_NUM_THREADS="3", MKL_NUM_THREADS="3", OPENBLAS_NUM_THREADS="3", PYTORCH_ALLOC_CONF="expandable_segments:True")
SETTLE_S = 120            # wait after a launch on a GPU before judging its memory again


def free_mb():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
    return {int(a): int(b) for a, b in (l.split(",") for l in out.strip().splitlines())}


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def main():
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    log("building caches")
    subprocess.run([PY, SCRIPT, "--prep"], check=True, stdout=open(os.path.join(ROOT, "logs", "prep.log"), "w"), stderr=subprocess.STDOUT)

    queue = deque()
    for n in NS:
        for seed in SEEDS:
            for m in METHODS:
                if not os.path.exists(os.path.join(ROOT, "results", f"{m}_N{n}_s{seed}.json")):
                    queue.append((m, n, seed, 1))
    total = len(queue)
    log(f"{total} jobs queued")

    running = {}            # Popen -> (job, gpu, start)
    last_launch = {g: 0.0 for g in GPUS}
    done = failed = 0
    while queue or running:
        for p in list(running):
            if p.poll() is not None:
                (m, n, seed, attempt), g, st = running.pop(p)
                if p.returncode == 0:
                    done += 1
                    log(f"DONE {m} N={n} s={seed} gpu{g} {time.time() - st:.0f}s  [{done}/{total}]")
                elif attempt < 2:
                    log(f"FAIL {m} N={n} s={seed} gpu{g} rc={p.returncode}; retrying")
                    queue.append((m, n, seed, attempt + 1))
                else:
                    failed += 1
                    log(f"FAIL {m} N={n} s={seed} gpu{g} rc={p.returncode}; giving up")
        if queue:
            fm = free_mb()
            for g in GPUS:
                if not queue:
                    break
                on_g = sum(1 for v in running.values() if v[1] == g)
                if on_g < MAX_PER_GPU and fm[g] >= MIN_FREE_MB and time.time() - last_launch[g] >= SETTLE_S:
                    m, n, seed, attempt = queue.popleft()
                    lf = open(os.path.join(ROOT, "logs", f"{m}_N{n}_s{seed}.log"), "a")
                    p = subprocess.Popen([PY, SCRIPT, "--method", m, "--n", n, "--seed", str(seed), "--device", f"cuda:{g}"],
                                         stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT, env=JOB_ENV)
                    running[p] = ((m, n, seed, attempt), g, time.time())
                    last_launch[g] = time.time()
                    log(f"START {m} N={n} s={seed} on gpu{g} (free {fm[g]} MB, {on_g + 1} jobs on gpu)")
        time.sleep(15)
    log(f"ALL FINISHED: {done} ok, {failed} failed")


if __name__ == "__main__":
    main()
