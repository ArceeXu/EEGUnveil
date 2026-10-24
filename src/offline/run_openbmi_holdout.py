import argparse
import csv
import gc
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__:
    from . import train_v65_distill as core
else:
    import train_v65_distill as core


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_split(report_path, split_path, split_seed):
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes.decode("utf-8"))
    digest = hashlib.sha256(report_bytes).hexdigest()

    if report.get("status") != "complete":
        raise ValueError("分组报告未完成")
    if report["config"].get("group_unit") != "subject":
        raise ValueError("需要按受试者统一分组的报告")

    members = {}
    for row in report["sessions"]:
        sid = int(row["subject_id"])
        group = row["group"]
        if group not in ("normal", "bci_blind"):
            raise ValueError(f"未知组别：{group}")
        if sid in members and members[sid] != group:
            raise ValueError(f"受试者 {sid} 两次 session 的组别不一致")
        members[sid] = group

    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if split["report_sha256"] != digest:
            raise ValueError("分组报告发生变化，禁止复用原划分")
        if split["split_seed"] != split_seed:
            raise ValueError("split seed 与已有划分不一致")
    else:
        rng = np.random.default_rng(split_seed)
        split = {
            "split_seed": split_seed,
            "report_sha256": digest,
            "train": [],
            "val": [],
            "test": [],
        }
        for group in ("normal", "bci_blind"):
            ids = np.array(sorted(
                sid for sid, label in members.items() if label == group
            ), dtype=int)
            if len(ids) < 5:
                raise ValueError(f"{group} 人数不足 5，需另行设计划分")
            rng.shuffle(ids)
            n = max(1, round(len(ids) * 0.2))
            split["test"].extend(ids[:n].tolist())
            split["val"].extend(ids[n:2*n].tolist())
            split["train"].extend(ids[2*n:].tolist())

        for key in ("train", "val", "test"):
            split[key].sort()
        split_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(split_path, split)

    tr, va, te = [set(split[k]) for k in ("train", "val", "test")]
    if tr & va or tr & te or va & te or tr | va | te != set(members):
        raise ValueError("受试者划分存在交叉、遗漏或未知成员")
    for key in ("train", "val", "test"):
        ids = split[key]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{key} 中存在重复受试者")
        if {members[sid] for sid in ids} != {"normal", "bci_blind"}:
            raise ValueError(f"{key} 必须同时包含正常组和 BCI 盲组")
    print("固定受试者划分：", split)
    return split


def load_partition(args, ids, phase="train"):
    return core.load_openbmi_groups(
        args.groups_dir,
        split=phase,
        subject_ids=ids,
        return_meta=True,
    )


def train(args, split):
    training_ids = list(split["train"])
    if args.train_population == "normal":
        report = json.loads((args.groups_dir / "report.json").read_text(encoding="utf-8"))
        normal_ids = {int(row["subject_id"]) for row in report["sessions"]
                      if row["group"] == "normal"}
        training_ids = sorted(set(training_ids) & normal_ids)
        if not training_ids:
            raise ValueError("训练划分中没有正常组受试者")
    print("实际参与参数训练的受试者：", training_ids, flush=True)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)

    core.ARTIFACTS_DIR = out
    core.SEED = args.seed
    core.BATCH_SIZE = args.batch_size
    core.EPOCHS = args.epochs
    core.NUM_WORKERS = args.num_workers
    core.SAMPLING_MODE = args.sampling
    core.MIX_LAMBDA = args.mix_lambda
    core.USE_DISTILLATION = not args.no_distillation

    # 本入口只运行一次固定训练/验证划分。
    # 禁用跨折全局停止及低准确率硬停止，保留验证集 patience 早停。
    core.GLOBAL_STOP_ACC = float("inf")
    core.HARD_STOP_EPOCH = args.epochs + 1
    core.GLOBAL_STOP_FLAG = False
    core.GLOBAL_BEST_ACC = -1.0
    core.GLOBAL_BEST_ARTIFACTS = {}

    core.RUN_CONFIG.update(
        protocol="fixed_subject_holdout",
        subject_split=split,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        sampling=args.sampling,
        use_distillation=not args.no_distillation,
        distill_alpha=core.DISTILL_ALPHA,
        distill_temperature=core.DISTILL_TEMP,
        variance_coefficient=core.LAMBDA_VAR,
        train_population=args.train_population,
        training_subject_ids=training_ids,
        checkpoint_selection="overall_validation_trial_accuracy",
        training_budget="epochs",
    )

    Xtr, ytr, gtr, str_, _ = load_partition(args, training_ids)
    if args.train_population == "normal" and not np.all(gtr == 0):
        raise ValueError("仅正常组训练意外包含其他组的数据")
    Xva, yva, gva, sva, _ = load_partition(args, split["val"])
    assert set(str_).isdisjoint(set(sva))

    seed_everything(args.seed)
    acc, labels, preds, report = core.train_fold_pipeline(
        Xtr, ytr, Xva, yva, 0, Xtr.shape, gtr, gva
    )
    core.save_global_artifacts()
    del Xtr, Xva
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    labels, preds = np.asarray(labels), np.asarray(preds)
    per_subject = [
        float(np.mean(preds[sva == sid] == labels[sva == sid]))
        for sid in np.unique(sva)
    ]
    write_json(out / "validation_metrics.json", {
        "trial_accuracy": float(acc),
        "subject_macro_accuracy": float(np.mean(per_subject)),
        "report": report,
    })
    write_json(out / "subject_split.json", split)
    print("训练结束；尚未读取测试阶段数据。")


def test(args, split):
    out = args.output_dir
    config = json.loads(
        (out / "model_config.json").read_text(encoding="utf-8")
    )
    if config["subject_split"] != split:
        raise ValueError("模型保存的划分与当前划分不一致")
    if (out / "test_metrics.json").exists():
        raise FileExistsError("该实验已有测试结果，避免意外覆盖")

    # 清理训练态配置；下面只做推理。
    core.RUN_CONFIG.update(config)
    core.RUN_CONFIG["data_root"] = str(args.data_root)

    X, y, groups, subjects, sessions = load_partition(
        args, split["test"], phase="test"
    )
    assert set(subjects).isdisjoint(
        set(split["train"]) | set(split["val"])
    )

    csp = core.joblib.load(out / "best_csp.pkl")
    scalers = core.joblib.load(out / "best_scalers.pkl")
    features = csp.transform(X)  # 测试时禁止 fit/fit_transform

    dataset = core.EEGDataset(X, y, features, scalers)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    model = core.EEGNetLight(
        n_channels=config["n_channels"],
        n_times=config["n_times"],
        n_classes=config["n_classes"],
        csp_dim=config["csp_dim"],
    ).to(core.DEVICE)
    model.load_state_dict(torch.load(
        out / "best_model.pth",
        map_location=core.DEVICE,
        weights_only=True,
    ))
    model.eval()

    probabilities = []
    with torch.inference_mode():
        for xb, _, cf in loader:
            logits = model(xb.to(core.DEVICE), cf.to(core.DEVICE))
            probabilities.append(logits.softmax(1).cpu().numpy())
    probabilities = np.concatenate(probabilities)
    pred = probabilities.argmax(1)

    with (out / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "subject_id", "session_id", "group",
            "y_true", "y_pred", "p_left", "p_right",
        ])
        writer.writerows(
            (int(sid), int(ses), "normal" if g == 0 else "bci_blind",
             int(yt), int(yp), float(p[0]), float(p[1]))
            for sid, ses, g, yt, yp, p in zip(
                subjects, sessions, groups, y, pred, probabilities
            )
        )

    rows = []
    for sid in np.unique(subjects):
        mask = subjects == sid
        rows.append({
            "subject_id": int(sid),
            "group": "normal" if groups[mask][0] == 0 else "bci_blind",
            "accuracy": float(np.mean(y[mask] == pred[mask])),
            "n_trials": int(mask.sum()),
        })

    with (out / "subject_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "seed": config["seed"],
        "sampling": config["sampling"],
        "use_distillation": config["use_distillation"],
        "effective_mix_lambda": config["effective_mix_lambda"],
        "trial_accuracy": float(np.mean(pred == y)),
        "subject_macro_accuracy": float(np.mean([
            row["accuracy"] for row in rows
        ])),
        "group_subject_macro_accuracy": {
            group: float(np.mean([
                row["accuracy"] for row in rows if row["group"] == group
            ]))
            for group in ("normal", "bci_blind")
        },
    }
    write_json(out / "test_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    del model, X, dataset
    gc.collect()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["train", "test"], required=True)
    p.add_argument("--groups-dir", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("/openbmi_dataset/Data"))
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--sampling", choices=["natural", "mixed"], default="mixed")
    p.add_argument("--train-population", choices=["all", "normal"], default="all")
    p.add_argument("--mix-lambda", type=float, default=0.75)
    p.add_argument("--no-distillation", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    if args.stage == "train" and args.train_population == "normal" and args.sampling != "natural":
        p.error("仅正常组训练请使用 --sampling natural（单组内随机遍历）")

    if args.sampling == "mixed" and not 0.5 < args.mix_lambda < 1:
        p.error("要求 0.5 < mix_lambda < 1")
    if args.batch_size < 1 or args.epochs < 1 or args.num_workers < 0:
        p.error("batch-size、epochs 必须为正数，num-workers 不能为负数")
    if args.stage == "train" and args.sampling == "mixed":
        normal = 2 * round(args.batch_size * args.mix_lambda / 2)
        blind = args.batch_size - normal
        if args.batch_size % 2 or not 0 < blind < normal:
            p.error("批大小及比例取整后必须保留两个组，且正常组占多数")
    if args.stage == "test" and not args.split_file.is_file():
        p.error("测试必须复用训练时已保存的 split-file，不能重新创建划分")

    core.RUN_CONFIG["data_root"] = str(args.data_root)
    split = get_split(
        args.groups_dir / "report.json", args.split_file, args.split_seed
    )
    if args.stage == "train":
        train(args, split)
    else:
        test(args, split)


if __name__ == "__main__":
    main()
