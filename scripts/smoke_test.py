#!/usr/bin/env python3
"""
Pre-flight smoke test for the JudgeJack parallel full-scale launch.

Runs a 40-record, 1-epoch training job for BOTH judge types, on the SAME
two pinned GPUs and with the SAME command structure as the real run, to
catch:
  - bad paths / missing files
  - argument-parsing failures (e.g. --gradient_accumulation_steps rejected)
  - immediate OOM at batch_size=4 / grad_accum=2
  - wrong GPU actually receiving the job (CUDA_VISIBLE_DEVICES mapping)
  - checkpoint I/O failures on this filesystem

Does NOT test: sustained multi-hour GPU/thermal stability, or the
incremental HF push (that's on a 45-min cycle and untested by this).

Usage:
    ~/judgejack_run/py312env/bin/python smoke_test.py
"""
import os, sys, json, shutil, subprocess, time

WORKDIR = os.path.expanduser("~/judgejack_run")
REPO_DIR = f"{WORKDIR}/badjudge"
PY = "/home/jupyter-avbj-f874/.conda/envs/judgejack_py310/bin/python"

CLEAN_TRAIN_FULL = f"{WORKDIR}/prm800k_clean_train_full.json"
POISONED_TRAIN_FULL = f"{WORKDIR}/prm800k_poisoned_train_full.json"
MID_MATCHED_PAIRS = f"{WORKDIR}/prm800k/matched_pairs_mid.json"

# Must match the pins from the notebook's Section 3.
CLEAN_GPU = "1"
POISONED_GPU = "2"

REQUIRED = {
    "venv python": PY,
    "repo": REPO_DIR,
    "clean train data": CLEAN_TRAIN_FULL,
    "poisoned train data": POISONED_TRAIN_FULL,
    "mid matched pairs": MID_MATCHED_PAIRS,
}
missing = [name for name, path in REQUIRED.items() if not os.path.exists(path)]
if missing:
    print("Cannot run smoke test -- missing prerequisites:")
    for m in missing:
        print(f"  - {m}")
    print("\nRun the notebook up through Section 5 (poison construction) first.")
    sys.exit(1)


def slice_records(src_path, dst_path, n=40):
    with open(src_path) as f:
        records = json.load(f)
    with open(dst_path, "w") as f:
        json.dump(records[:n], f)
    return len(records[:n])


SMOKE_DIR = f"{WORKDIR}/smoke_test"
os.makedirs(SMOKE_DIR, exist_ok=True)
clean_smoke_data = f"{SMOKE_DIR}/clean_smoke.json"
poisoned_smoke_data = f"{SMOKE_DIR}/poisoned_smoke.json"
n_clean = slice_records(CLEAN_TRAIN_FULL, clean_smoke_data)
n_poisoned = slice_records(POISONED_TRAIN_FULL, poisoned_smoke_data)
print(f"Sliced {n_clean} clean / {n_poisoned} poisoned records for smoke test")


def launch(judge_type, train_data, gpu_id, out_dir):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    cmd = [
        PY, "-u", "-m", "src.pilot.train_judge",
        "--judge_type", judge_type,
        "--base_model", "Qwen/Qwen2.5-1.5B-Instruct",
        "--train_data", train_data,
        "--epochs", "1", "--lr", "2e-4", "--batch_size", "4",
        "--gradient_accumulation_steps", "2",
        "--probe_eval_data", MID_MATCHED_PAIRS,
        "--probe_every_n_steps", "3", "--probe_patience", "9999",
        "--out_dir", out_dir,
    ]
    log_path = f"{out_dir}.log"
    logfile = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT,
                             text=True, env=env, cwd=REPO_DIR)
    return proc, logfile, log_path


def gpu_mem(gpu_id):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout
    for line in out.strip().splitlines():
        idx, mem = [x.strip() for x in line.split(",")]
        if idx == gpu_id:
            return int(mem)
    return 0


clean_out = f"{SMOKE_DIR}/clean_out"
poisoned_out = f"{SMOKE_DIR}/poisoned_out"

print(f"\nLaunching smoke runs -- clean on GPU {CLEAN_GPU}, poisoned on GPU {POISONED_GPU}...")
clean_proc, clean_log, clean_log_path = launch("clean", clean_smoke_data, CLEAN_GPU, clean_out)
poisoned_proc, poisoned_log, poisoned_log_path = launch("poisoned", poisoned_smoke_data, POISONED_GPU, poisoned_out)

clean_gpu_confirmed = False
poisoned_gpu_confirmed = False
start = time.time()
TIMEOUT_S = 600  # 10 min ceiling

while (clean_proc.poll() is None or poisoned_proc.poll() is None) and (time.time() - start) < TIMEOUT_S:
    time.sleep(10)
    elapsed = time.time() - start
    if not clean_gpu_confirmed and gpu_mem(CLEAN_GPU) > 500:
        clean_gpu_confirmed = True
    if not poisoned_gpu_confirmed and gpu_mem(POISONED_GPU) > 500:
        poisoned_gpu_confirmed = True
    print(f"[{elapsed:.0f}s] clean:{'done' if clean_proc.poll() is not None else 'running'} "
          f"(GPU{CLEAN_GPU} active:{clean_gpu_confirmed}) | "
          f"poisoned:{'done' if poisoned_proc.poll() is not None else 'running'} "
          f"(GPU{POISONED_GPU} active:{poisoned_gpu_confirmed})")

clean_log.close()
poisoned_log.close()

if clean_proc.poll() is None or poisoned_proc.poll() is None:
    print("\nTIMEOUT -- one or both smoke runs didn't finish in 10 min. Likely stuck.")
    clean_proc.kill()
    poisoned_proc.kill()
    sys.exit(1)

print(f"\nClean smoke run exit code: {clean_proc.returncode}")
print(f"Poisoned smoke run exit code: {poisoned_proc.returncode}")

ok = True
for label, code, log_path, out_dir, gpu_confirmed in [
    ("clean", clean_proc.returncode, clean_log_path, clean_out, clean_gpu_confirmed),
    ("poisoned", poisoned_proc.returncode, poisoned_log_path, poisoned_out, poisoned_gpu_confirmed),
]:
    if code != 0:
        ok = False
        print(f"\n--- {label} FAILED -- tail of log ---")
        with open(log_path) as f:
            print("".join(f.readlines()[-30:]))
        continue
    if not gpu_confirmed:
        ok = False
        print(f"\n{label}: exited 0 but its pinned GPU never showed memory usage -- "
              f"CUDA_VISIBLE_DEVICES pinning may not be working as expected")
        continue
    ckpts = os.listdir(out_dir) if os.path.isdir(out_dir) else []
    if not any(d.startswith("checkpoint-") for d in ckpts):
        ok = False
        print(f"\n{label}: exited 0 but no checkpoint directory was created")
        continue
    print(f"{label}: OK -- correct GPU used, checkpoint created")

print("\n" + ("ALL CHECKS PASSED -- safe to launch the real run without a 45-60min check-in."
              if ok else
              "SOMETHING FAILED -- do not launch the full run yet. Read the output above."))

if ok:
    shutil.rmtree(SMOKE_DIR)
    print("Cleaned up smoke test artifacts.")
else:
    print(f"Smoke test artifacts left in place for debugging: {SMOKE_DIR}")
