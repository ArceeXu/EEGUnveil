"""Linux-only migration from a paused single-worker lambda scheduler to two slots.

The existing training process occupies one slot until it exits. The original
scheduler and its lock remain paused/intact; never resume that scheduler.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__:
    from . import search_openbmi_lambda as search
else:
    import search_openbmi_lambda as search


def parse_stat(text):
    fields = text[text.rfind(")") + 2:].split()
    return {"state": fields[0], "ppid": int(fields[1]), "start": fields[19]}


def process(pid):
    try:
        root = Path("/proc") / str(pid)
        info = parse_stat((root / "stat").read_text())
        info["argv"] = [p.decode() for p in (root / "cmdline").read_bytes().split(b"\0") if p]
        return info
    except (FileNotFoundError, ProcessLookupError):
        return None


def ended(current, original):
    # A paused parent cannot reap its child: zombies must count as finished.
    return current is None or current["start"] != original["start"] or current["state"] == "Z"


def argument(argv, key):
    return argv[argv.index(key) + 1]


def verify_scheduler(args, identity, require_paused=True):
    info = process(args.scheduler_pid)
    if info is None or info["start"] != identity["start"]:
        raise RuntimeError("原搜索调度器身份发生变化，停止启动后续任务")
    if require_paused and info["state"] not in ("T", "t"):
        raise RuntimeError("原搜索调度器没有暂停；禁止同时调度同一批实验")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("check", "train"), required=True)
    p.add_argument("--scheduler-pid", type=int, required=True)
    p.add_argument("--training-pid", type=int, required=True)
    p.add_argument("--groups-dir", type=Path, default=Path("/openbmi_dataset/Data/OpenBMI/groups/csp70_20"))
    p.add_argument("--data-root", type=Path, default=Path("/openbmi_dataset/Data"))
    p.add_argument("--split-file", type=Path, default=Path("outputs/splits/openbmi_subject_holdout.json"))
    p.add_argument("--baseline-root", type=Path, default=Path("models/offline/holdout"))
    p.add_argument("--search-root", type=Path, default=Path("models/offline/lambda_analysis"))
    args = p.parse_args()
    args.allow_incomplete_baselines = True
    split = search.preflight(args)
    scheduler, child = process(args.scheduler_pid), process(args.training_pid)
    if not scheduler or not any(Path(a).name == "search_openbmi_lambda.py" for a in scheduler["argv"]):
        raise RuntimeError("指定 PID 不是原 λ 搜索调度器，请重新查询进程")
    if not child or child["state"] == "Z" or child["ppid"] != args.scheduler_pid:
        raise RuntimeError("指定训练进程已结束或不属于该调度器，请重新查询进程")
    argv = child["argv"]
    if not any(Path(a).name == "run_openbmi_holdout.py" for a in argv):
        raise RuntimeError("指定 PID 不是训练入口")
    if argument(argv, "--stage") != "train" or "--no-distillation" in argv:
        raise RuntimeError("接管目标必须是蒸馏训练")
    lam, seed = float(argument(argv, "--mix-lambda")), int(argument(argv, "--seed"))
    if lam not in search.LAMBDAS or lam == .75 or seed not in search.SEEDS:
        raise RuntimeError("当前训练不属于新增 λ 搜索任务")
    adopted = search.result_dir(args, lam, seed)
    child_cwd = (Path("/proc") / str(args.training_pid) / "cwd").resolve()
    source_output = Path(argument(argv, "--output-dir"))
    if not source_output.is_absolute():
        source_output = child_cwd / source_output
    if source_output.resolve() != adopted.resolve():
        raise RuntimeError("当前训练输出目录不匹配")
    old_lock = args.search_root / ".search.lock"
    if old_lock.read_text().strip() != str(args.scheduler_pid):
        raise RuntimeError("原搜索锁与指定调度器不一致")

    jobs = [(lam, seed, adopted, True)]
    for candidate in search.LAMBDAS:
        if candidate == .75:
            continue
        for s in search.SEEDS:
            folder = search.result_dir(args, candidate, s)
            if folder == adopted:
                continue
            if folder.exists():
                search.validate_result(folder, split, candidate, s)
            else:
                jobs.append((candidate, s, folder, False))
    print(f"接管 λ={lam}, seed={seed}, PID={args.training_pid}；另有 {len(jobs)-1} 项待训练", flush=True)
    if args.stage == "check":
        print("检查通过。只暂停原调度器后，再以 --stage train 启动。", flush=True)
        return

    verify_scheduler(args, scheduler)
    # flock is released automatically on exit. Keep the old lock untouched.
    import fcntl
    with (args.search_root / ".takeover.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("接管调度器已运行，禁止重复启动")
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        logs = args.search_root / "logs"
        logs.mkdir(exist_ok=True)

        def run_one(job):
            candidate, s, folder, is_adopted = job
            verify_scheduler(args, scheduler)
            if is_adopted:
                print(f"保留现有训练 PID={args.training_pid}，占用第一个搜索名额", flush=True)
                while not ended(process(args.training_pid), child):
                    verify_scheduler(args, scheduler)
                    time.sleep(10)
            else:
                command = [sys.executable, "-u", str(Path(__file__).with_name("run_openbmi_holdout.py")),
                           "--stage", "train", "--groups-dir", str(args.groups_dir),
                           "--data-root", str(args.data_root), "--split-file", str(args.split_file),
                           "--split-seed", "2026", "--sampling", "mixed", "--mix-lambda", str(candidate),
                           "--seed", str(s), "--batch-size", "32", "--epochs", "120", "--num-workers", "0",
                           "--output-dir", str(folder)]
                print(f"开始 λ={candidate}, seed={s}", flush=True)
                with (logs / f"lambda{candidate:g}_seed{s}.log").open("a", encoding="utf-8") as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, check=True)
            search.validate_result(folder, split, candidate, s)
            print(f"完成 λ={candidate}, seed={s}", flush=True)

        with ThreadPoolExecutor(max_workers=2) as pool:
            iterator = iter(jobs)
            initial = [next(iterator, None) for _ in range(2)]
            active = {pool.submit(run_one, job) for job in initial if job is not None}
            while active:
                future = next(as_completed(active))
                active.remove(future)
                future.result()
                job = next(iterator, None)
                if job is not None:
                    active.add(pool.submit(run_one, job))
        print("新增 λ 搜索全部完成。原调度器仍保持暂停，禁止恢复。", flush=True)


if __name__ == "__main__":
    main()
