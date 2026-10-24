"""Fixed-grid lambda search: reuse C=0.75 and select using validation only.

Run from the project root. This script never invokes the held-out test stage.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


LAMBDAS = (0.5625, 0.625, 0.75, 0.875, 0.9375)
SEEDS = (42, 2024, 3407)
ARTIFACTS = (
    "best_model.pth", "best_csp.pkl", "best_scalers.pkl",
    "model_config.json", "validation_metrics.json", "subject_split.json",
)
PROTOCOL = {
    "protocol": "fixed_subject_holdout", "batch_size": 32, "epochs": 120,
    "sampling": "mixed", "use_distillation": True,
    "distill_alpha": 0.5, "distill_temperature": 3.0,
    "variance_coefficient": 1e-5,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def complete(folder):
    return all((folder / name).is_file() and (folder / name).stat().st_size > 0
               for name in ARTIFACTS)


def result_dir(args, lam, seed):
    if lam == 0.75:
        return args.baseline_root / f"C_seed{seed}"
    return args.search_root / f"lambda{lam:g}_seed{seed}"


def validate_result(folder, split, lam, seed):
    if not complete(folder):
        raise ValueError(f"实验结果缺失或不完整：{folder}")
    config = read_json(folder / "model_config.json")
    expected = dict(PROTOCOL, seed=seed, mix_lambda=lam, effective_mix_lambda=lam)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{folder}: {key}={config.get(key)!r}，预期 {value!r}")
    if config.get("subject_split") != split or read_json(folder / "subject_split.json") != split:
        raise ValueError(f"受试者划分不一致：{folder}")
    metrics = read_json(folder / "validation_metrics.json")
    normal = float(metrics["report"]["group_accuracy"]["normal"])
    blind = float(metrics["report"]["group_accuracy"]["bci_blind"])
    overall = float(metrics["subject_macro_accuracy"])
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (normal, blind, overall)):
        raise ValueError(f"验证指标无效：{folder}")
    return {
        "lambda": lam, "seed": seed, "normal": normal, "bci_blind": blind,
        "overall": overall, "score": (normal + blind) / 2,
        "result_dir": str(folder),
    }


def preflight(args):
    split = read_json(args.split_file)  # Missing split is an error, never recreate it.
    report_bytes = (args.groups_dir / "report.json").read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != split["report_sha256"]:
        raise ValueError("分组报告与原划分的 SHA256 不一致")
    if split["split_seed"] != 2026:
        raise ValueError("本次搜索要求沿用 split-seed=2026")
    parts = [set(split[k]) for k in ("train", "val", "test")]
    if any(len(split[k]) != len(set(split[k])) for k in ("train", "val", "test")):
        raise ValueError("受试者划分含重复成员")
    if parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2]:
        raise ValueError("训练、验证、测试受试者有交叉")
    report = json.loads(report_bytes)
    if report.get("status") != "complete" or report["config"].get("group_unit") != "subject":
        raise ValueError("需要完整的受试者级分组报告")

    # Existing validation group metrics pool trials. They equal subject means
    # only when validation subjects within each group have equal trial counts.
    counts, memberships = {}, {}
    for row in report["sessions"]:
        sid, group = int(row["subject_id"]), row["group"]
        if sid in memberships and memberships[sid] != group:
            raise ValueError("同一受试者的组别不一致")
        memberships[sid] = group
        if sid in parts[1]:
            counts[sid] = counts.get(sid, 0) + int(row["trials"])
    if set.union(*parts) != set(memberships):
        raise ValueError("划分与报告中的受试者不一致")
    for group in ("normal", "bci_blind"):
        values = [counts[sid] for sid in parts[1] if memberships[sid] == group]
        if not values or len(set(values)) != 1 or values[0] <= 0:
            raise ValueError("组内验证试次数不相等，不能将 trial 指标当作受试者平均指标")

    early = args.allow_incomplete_baselines and args.stage in ("check", "train")
    completed_c = 0
    for seed in SEEDS:
        for method in "ABCD":
            folder = args.baseline_root / f"{method}_seed{seed}"
            if not complete(folder):
                if early:
                    print(f"原实验尚未完成，交由原调度器继续：{folder}", flush=True)
                    continue
                raise ValueError(f"请先完成原四组实验：{folder}")
            if read_json(folder / "subject_split.json") != split:
                raise ValueError(f"原实验划分不一致：{folder}")
        c_folder = result_dir(args, 0.75, seed)
        if complete(c_folder):
            validate_result(c_folder, split, 0.75, seed)
            completed_c += 1
    if early and not completed_c:
        raise ValueError("提前搜索至少需要一份完整 C 组结果，以核对实际训练协议")
    return split


def planned_jobs(args, split):
    pending = []
    for lam in LAMBDAS:
        for seed in SEEDS:
            folder = result_dir(args, lam, seed)
            if lam == 0.75:
                # Never schedule baseline C jobs here, including when absent.
                if complete(folder):
                    validate_result(folder, split, lam, seed)
                    print(f"复用已完成结果：{folder}", flush=True)
                else:
                    print(f"等待原 C 组完成，不重复启动：{folder}", flush=True)
                continue
            if folder.exists():
                validate_result(folder, split, lam, seed)
                print(f"复用已完成结果：{folder}", flush=True)
            else:
                pending.append((lam, seed, folder))
    return pending


def run_training(args, split):
    args.search_root.mkdir(parents=True, exist_ok=True)
    lock = args.search_root / ".search.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(f"已有搜索锁：{lock}。勿重复启动；异常中断后需先确认旧任务已退出。")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(str(os.getpid()))
        jobs = planned_jobs(args, split)
        logs = args.search_root / "logs"
        logs.mkdir(exist_ok=True)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        script = Path(__file__).with_name("run_openbmi_holdout.py")

        def run_one(job):
            lam, seed, folder = job
            log = logs / f"lambda{lam:g}_seed{seed}.log"
            command = [
                sys.executable, "-u", str(script), "--stage", "train",
                "--groups-dir", str(args.groups_dir), "--data-root", str(args.data_root),
                "--split-file", str(args.split_file), "--split-seed", "2026",
                "--sampling", "mixed", "--mix-lambda", str(lam), "--seed", str(seed),
                "--batch-size", "32", "--epochs", "120", "--num-workers", "0",
                "--output-dir", str(folder),
            ]
            print(f"[{datetime.now().isoformat(timespec='seconds')}] 开始 λ={lam}, seed={seed}; 日志：{log}", flush=True)
            with log.open("a", encoding="utf-8") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, check=True)
            validate_result(folder, split, lam, seed)
            print(f"完成 λ={lam}, seed={seed}", flush=True)

        # Submit only as slots become free. On failure, finish already-running
        # jobs without scheduling further experiments.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            iterator = iter(jobs)
            active = {pool.submit(run_one, job) for job in [next(iterator, None) for _ in range(args.workers)] if job is not None}
            while active:
                done = next(as_completed(active))
                active.remove(done)
                done.result()
                job = next(iterator, None)
                if job is not None:
                    active.add(pool.submit(run_one, job))
        print("λ 搜索训练全部完成；尚未读取最终测试数据。", flush=True)
    finally:
        lock.unlink()


def aggregate(rows):
    summaries = []
    for lam in LAMBDAS:
        selected = [row for row in rows if row["lambda"] == lam]
        if sorted(row["seed"] for row in selected) != sorted(SEEDS):
            raise ValueError(f"λ={lam} 的三个 seed 不完整或重复")
        summary = {"lambda": lam}
        for metric in ("normal", "bci_blind", "overall", "score"):
            values = [row[metric] for row in selected]
            summary[metric + "_mean"] = statistics.mean(values)
            summary[metric + "_std"] = statistics.stdev(values)
        summaries.append(summary)
    # Predeclared tie rule: exactly equal means prefer lambda closest to 0.75,
    # then the smaller lambda. Do not select a single lucky seed.
    best = min(summaries, key=lambda r: (-r["score_mean"], abs(r["lambda"] - .75), r["lambda"]))
    return summaries, best


def summarize(args, split):
    rows = [validate_result(result_dir(args, lam, seed), split, lam, seed)
            for lam in LAMBDAS for seed in SEEDS]
    summaries, best = aggregate(rows)
    if args.font is None or not args.font.is_file():
        raise ValueError("汇总绘图需指定存在的宋体文件 --font assets/fonts/simsun.ttc")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(args.font))
    font_name = font_manager.FontProperties(fname=str(args.font)).get_name()
    plt.rcParams.update({"font.family": font_name, "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 11})
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in (("lambda_validation_runs.csv", rows), ("lambda_validation_summary.csv", summaries)):
        with (args.summary_dir / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    selection = {
        "selected_lambda": best["lambda"], "criterion": "mean across seeds of (normal + bci_blind)/2",
        "tie_rule": "closest to 0.75, then smaller lambda",
        "epoch_selection": "existing overall validation trial accuracy",
        "score_mean": best["score_mean"], "score_std": best["score_std"],
        "seeds": SEEDS, "candidates": LAMBDAS, "subject_split": split,
        "metric_units": "fraction in [0,1]; standard deviation across three training seeds",
        "scope": "best on this validation split among this grid, not a guaranteed global optimum",
        "test_used": False,
    }
    (args.summary_dir / "selected_lambda.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for metric, label, color, marker in (
        ("normal", "正常组", "#588896", "o"),
        ("bci_blind", "BCI盲组", "#63b8ac", "s"),
        ("score", "两组等权平均（选择指标）", "#72a8b5", "D"),
        ("overall", "全体验证受试者平均", "#8ed2c7", "^"),
    ):
        ax.errorbar(LAMBDAS, [100*r[metric+"_mean"] for r in summaries],
                    yerr=[100*r[metric+"_std"] for r in summaries],
                    label=label, color=color, marker=marker, capsize=3)
    ax.axvline(best["lambda"], color="#588896", linestyle="--", alpha=.5)
    ax.set(xlabel="正常组采样占比 λ", ylabel="验证准确率（%）", ylim=(0, 100),
           title="混合采样比例对留出验证受试者的影响")
    ax.set_xticks(LAMBDAS)
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=9)
    fig.text(.5, .01, "误差线为三个训练随机种子间的样本标准差；未使用最终测试集", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(args.summary_dir / f"lambda_validation.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"验证集选定 λ={best['lambda']}; S={best['score_mean']:.4f} ± {best['score_std']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "train", "summarize"), required=True)
    parser.add_argument("--groups-dir", type=Path, default=Path("/openbmi_dataset/Data/OpenBMI/groups/csp70_20"))
    parser.add_argument("--data-root", type=Path, default=Path("/openbmi_dataset/Data"))
    parser.add_argument("--split-file", type=Path, default=Path("outputs/splits/openbmi_subject_holdout.json"))
    parser.add_argument("--baseline-root", type=Path, default=Path("models/offline/holdout"))
    parser.add_argument("--search-root", type=Path, default=Path("models/offline/lambda_analysis"))
    parser.add_argument("--summary-dir", type=Path, default=Path("outputs/figures/lambda_analysis"))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--allow-incomplete-baselines", action="store_true",
                        help="Allow candidate training before all baselines finish; limit total concurrent GPU jobs externally")
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    if args.allow_incomplete_baselines:
        if args.stage == "summarize":
            parser.error("汇总不能放宽完整性检查，请去掉 --allow-incomplete-baselines")
    split = preflight(args)
    if args.stage == "check":
        pending = planned_jobs(args, split)
        print(f"检查通过，尚需训练 {len(pending)} 次；固定采用两组等权验证指标选择 λ。")
    elif args.stage == "train":
        run_training(args, split)
    else:
        summarize(args, split)


if __name__ == "__main__":
    main()
