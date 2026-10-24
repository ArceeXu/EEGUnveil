"""Linux only: drain an existing search's children, then restart with two workers.

Run in the same project directory as the original search, inside tmux.
Never signals training children. A failure before retiring the old scheduler
resumes it. Requires the standard paths used in search_openbmi_lambda.py.
"""
import argparse
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

if __package__:
    from . import search_openbmi_lambda as search
else:
    import search_openbmi_lambda as search


def process(pid):
    try:
        root = Path(f"/proc/{pid}")
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        argv = (root / "cmdline").read_bytes().decode().strip("\0").split("\0")
        return dict(state=fields[0], ppid=int(fields[1]), start=fields[19], argv=argv)
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(pid, start):
    info = process(pid)
    return info if info and info["start"] == start else None


def argument(argv, name):
    return argv[argv.index(name) + 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("此切换脚本仅在 Linux 服务器运行")
    import fcntl
    root = Path.cwd().resolve()
    search_root = root / "models/offline/lambda_analysis"
    lock = search_root / ".search.lock"
    # Separate migration lock prevents two upgrade commands operating together.
    with (search_root / ".upgrade.lock").open("a") as upgrade_lock:
        fcntl.flock(upgrade_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pid = int(lock.read_text().strip())
        old = process(pid)
        if not old or not any(Path(a).name == "search_openbmi_lambda.py" for a in old["argv"]):
            raise RuntimeError("搜索锁未对应正在运行的搜索器，未执行切换")
        if Path(f"/proc/{pid}/cwd").resolve() != root:
            raise RuntimeError("请在原搜索器的工作目录运行")
        if argument(old["argv"], "--stage") != "train":
            raise RuntimeError("目标不是训练调度器")
        allowed = {"--stage", "--workers", "--allow-incomplete-baselines"}
        if any(a.startswith("--") and a not in allowed for a in old["argv"]):
            raise RuntimeError("原搜索器使用自定义路径或参数，需单独检查，未执行切换")
        args = SimpleNamespace(
            allow_incomplete_baselines=True, stage="train",
            split_file=root / "outputs/splits/openbmi_subject_holdout.json",
            groups_dir=Path("/openbmi_dataset/Data/OpenBMI/groups/csp70_20"),
            baseline_root=root / "models/offline/holdout", search_root=search_root,
        )
        split = search.preflight(args)
        retired = False
        try:
            os.kill(pid, signal.SIGSTOP)
            for _ in range(100):
                info = same_process(pid, old["start"])
                if info and info["state"] == "T":
                    break
                time.sleep(.1)
            else:
                raise RuntimeError("未能确认调度器已暂停")
            print(f"旧搜索器 {pid} 已暂停调度；现有训练继续。", flush=True)
            children = []
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit():
                    continue
                child_pid = int(entry.name)
                child = process(child_pid)
                if not child or child["ppid"] != pid:
                    continue
                if child["state"] == "Z":
                    continue
                argv = child["argv"]
                if not any(Path(a).name == "run_openbmi_holdout.py" for a in argv):
                    raise RuntimeError(f"遇到未知子进程 {child_pid}，取消切换")
                folder = (root / argument(argv, "--output-dir")).resolve()
                folder.relative_to(search_root)
                children.append((child_pid, child["start"], folder,
                                 float(argument(argv, "--mix-lambda")),
                                 int(argument(argv, "--seed"))))
                print(f"等待现有训练 {child_pid} 完成：{folder.name}", flush=True)
            while any((info := same_process(p, start)) and info["state"] != "Z"
                      for p, start, *_ in children):
                time.sleep(10)
            for _, _, folder, lam, seed in children:
                search.validate_result(folder, split, lam, seed)
            # Reject any other incomplete candidate before retiring the manager.
            search.planned_jobs(args, split)
            info = same_process(pid, old["start"])
            if not info or info["state"] != "T":
                raise RuntimeError("旧搜索器状态变化，取消切换")
            if int(lock.read_text().strip()) != pid:
                raise RuntimeError("搜索锁归属变化，取消切换")
            # All training children have exited and their outputs were validated.
            # Kill only the stopped scheduler, never its process group/children.
            os.kill(pid, signal.SIGKILL)
            retired = True
            for _ in range(100):
                info = same_process(pid, old["start"])
                if not info or info["state"] == "Z":
                    break
                time.sleep(.1)
            else:
                raise RuntimeError("旧搜索器尚未退出，保留搜索锁")
            if int(lock.read_text().strip()) != pid:
                raise RuntimeError("锁归属变化，保留搜索锁")
            lock.unlink()
        finally:
            if not retired and same_process(pid, old["start"]):
                os.kill(pid, signal.SIGCONT)
                print("未完成切换，已恢复原搜索器。", flush=True)
        print("当前训练已保存；启动两路 λ 搜索，自动跳过完整结果。", flush=True)
        script = Path(__file__).with_name("search_openbmi_lambda.py").resolve()
        os.execv(sys.executable, [sys.executable, "-u", str(script), "--stage", "train",
                                 "--allow-incomplete-baselines", "--workers", "2"])


if __name__ == "__main__":
    main()
