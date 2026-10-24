"""Group official OpenBMI training runs by train-only CSP+LDA CV accuracy."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import mne
import numpy as np
import scipy
import sklearn
from mne.decoding import CSP
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[2]
MOTOR_CHANNELS = [
    "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
]
TMIN, TMAX = 1.0, 3.5
FILTER_ORDER = 5


def group_name(accuracy, threshold=0.70):
    if not 0 <= accuracy <= 1 or not 0 < threshold < 1:
        raise ValueError("Accuracy/threshold must be finite fractions in [0, 1]")
    return "bci_blind" if accuracy < threshold else "normal"


def load_training(path):
    """Only open train.npz; reject pooled official train/test data."""
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    config = json.loads(str(data["metadata_json"].item()))
    if config["mode"] != "official" or not np.all(data["source_run"] == "EEG_MI_train"):
        raise ValueError(f"{path}: only official training-phase data are supported")
    return data


def prepare_scoring(data):
    """Select motor channels and locate the cue-relative window in saved epochs."""
    names = [str(name).strip().casefold() for name in data["channel_names"]]
    picks = [names.index(name.casefold()) for name in MOTOR_CHANNELS]
    config = json.loads(str(data["metadata_json"].item()))
    sfreq = float(data["sfreq"])
    # Account for an input split that already starts later than cue onset.
    origin = float(config["tmin"])
    start = round((TMIN - origin) * sfreq)
    stop = round((TMAX - origin) * sfreq)
    if start < 0 or stop > data["X"].shape[-1] or stop <= start:
        raise ValueError("Input epochs must cover cue-relative [1, 3.5) seconds")
    return data["X"][:, picks, :], slice(start, stop)


def score_training(X, y, sfreq, folds=5, seed=42, components=4,
                   low=8.0, high=30.0, epoch_slice=None):
    """Each trial gets exactly one held-out prediction; refit CSP in each fold."""
    sos = butter(FILTER_ORDER, [low, high], btype="bandpass", fs=sfreq, output="sos")
    # Filter the available epoch first, then crop to reduce crop-edge artifacts.
    filtered = sosfiltfilt(sos, X.astype(np.float64), axis=-1)
    if epoch_slice is not None:
        filtered = filtered[..., epoch_slice]
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    prediction = np.full(len(y), -1, dtype=np.int64)
    fold_id = np.full(len(y), -1, dtype=np.int64)
    scores = []
    for fold, (train, valid) in enumerate(cv.split(filtered, y)):
        pipeline = make_pipeline(
            CSP(n_components=components, reg=0.1, log=True,
                norm_trace=False, cov_est="concat", rank="full",
                component_order="alternate"),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        )
        with mne.use_log_level("ERROR"):
            pipeline.fit(filtered[train], y[train])
            prediction[valid] = pipeline.predict(filtered[valid])
        fold_id[valid] = fold
        scores.append(float(np.mean(prediction[valid] == y[valid])))
    return float(np.mean(prediction == y)), scores, prediction, fold_id


def process_subject(paths, output, args):
    """分别评估两个 session，再按平均准确率给受试者统一分组。"""
    results = []
    subject = None
    seen_sessions = set()
    for path in sorted(paths):
        data = load_training(path)
        subject_ids = np.unique(data["subject_id"])
        session_ids = np.unique(data["session_id"])
        current_subject = int(subject_ids[0])
        session = int(session_ids[0])
        if subject is None:
            subject = current_subject
        if len(subject_ids) != 1 or len(session_ids) != 1 or current_subject != subject or session in seen_sessions:
            raise ValueError("Expected two distinct sessions from the same subject")
        seen_sessions.add(session)
        print(
            f"Scoring sub-{subject:02d}/ses-{session:02d} ...",
            flush=True,
        )
        scoring_X, epoch_slice = prepare_scoring(data)
        accuracy, scores, prediction, fold_id = score_training(
            scoring_X,
            data["y"],
            float(data["sfreq"]),
            args.folds,
            args.seed,
            args.components,
            args.low,
            args.high,
            epoch_slice=epoch_slice,
        )
        results.append({
            "path": path,
            "data": data,
            "session": session,
            "accuracy": accuracy,
            "scores": scores,
            "prediction": prediction,
            "fold_id": fold_id,
        })
        print(f"  Session CV={accuracy:.2%}", flush=True)
    if seen_sessions != {1, 2}:
        raise ValueError(
            f"受试者 {subject}: 需要 session 1 和 2，"
            f"实际为 {sorted(seen_sessions)}"
        )
    subject_accuracy = float(np.mean([
        result["accuracy"] for result in results
    ]))
    group = group_name(subject_accuracy, args.threshold)
    print(
        f"Subject sub-{subject:02d}: "
        f"mean CV={subject_accuracy:.2%}, group={group}",
        flush=True,
    )
    session_rows = []
    for result in results:
        path = result["path"]
        data = result["data"]
        session = result["session"]

        folder = (
            output
            / group
            / f"sub-{subject:02d}"
            / f"ses-{session:02d}"
        )
        folder.mkdir(parents=True, exist_ok=False)
        data.update(
            group_label=np.full(
                len(data["y"]),
                int(group == "bci_blind"),
                dtype=np.int64,
            ),
            group_name=np.array(group),
            csp_cv_accuracy=np.array(result["accuracy"]),
            subject_csp_cv_accuracy=np.array(subject_accuracy),
            # The exported X stays unchanged; these fields describe CSP scoring.
            csp_channel_names=np.array(MOTOR_CHANNELS),
            csp_time_window=np.array([TMIN, TMAX]),
            csp_oof_prediction=result["prediction"],
            csp_fold_id=result["fold_id"],
        )
        with (folder / "train.npz").open("xb") as stream:
            np.savez_compressed(stream, **data)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        session_rows.append({
            "subject_id": subject,
            "session_id": session,
            "group": group,
            "cv_accuracy": result["accuracy"],
            "subject_cv_accuracy": subject_accuracy,
            "fold_accuracies": result["scores"],
            "trials": len(data["y"]),
            "left": int(np.sum(data["y"] == 0)),
            "right": int(np.sum(data["y"] == 1)),
            "source_train": str(path.resolve()),
            "source_sha256": digest.hexdigest(),
            "grouped_train": str(
                (folder / "train.npz").relative_to(output)
            ),
        })
    accuracy_by_session = {
        result["session"]: result["accuracy"]
        for result in results
    }
    subject_row = {
        "subject_id": subject,
        "session1_cv_accuracy": accuracy_by_session[1],
        "session2_cv_accuracy": accuracy_by_session[2],
        "mean_cv_accuracy": subject_accuracy,
        "group": group,
        "trials": sum(row["trials"] for row in session_rows),
    }
    return subject_row, session_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "Data/OpenBMI/splits/official_250hz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Data/OpenBMI/groups/csp70")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--components", type=int, choices=(4,), default=4,
                        help="Two CSP components from each eigenvalue extreme")
    parser.add_argument("--low", type=float, default=8.0)
    parser.add_argument("--high", type=float, default=30.0)
    args = parser.parse_args()
    files = sorted(args.split_dir.glob("sub-*/ses-*/train.npz"))
    if not files:
        parser.error(f"No training splits found under {args.split_dir}")
    subject_files = {}
    for path in files:
        subject_folder = path.parent.parent.name
        subject_files.setdefault(subject_folder, []).append(path)

    for subject_folder, paths in sorted(subject_files.items()):
        session_folders = {path.parent.name for path in paths}
        if len(paths) != 2 or session_folders != {"ses-01", "ses-02"}:
            parser.error(f"{subject_folder}: both session 1 and session 2 are required")

    args.output_dir.mkdir(parents=True)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    config.update(
        group_unit="subject",
        aggregation="arithmetic mean of session 1 and session 2 CV accuracies",
        required_sessions=[1, 2],
        boundary="mean accuracy < threshold => bci_blind",
        channels=MOTOR_CHANNELS,
        tmin=TMIN,
        tmax=TMAX,
        filter="trialwise Butterworth order 5 sosfiltfilt, then crop",
        filter_order=FILTER_ORDER,
        csp_component_order="alternate",
        exported_X="unchanged input; motor channels and time window apply to scoring only",
        csp_reg=0.1,
        csp_rank="full",
        lda="lsqr, shrinkage=auto",
        versions={
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "mne": mne.__version__,
        },
    )
    report = {
        "config": config,
        "status": "incomplete",
        "subjects": [],  # 每个人一条记录，用于统计人数。
        "sessions": [],  # 每个 session 一条记录，用于追溯数据。
    }

    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for subject_folder, paths in sorted(subject_files.items()):
        subject_row, session_rows = process_subject(
            paths, args.output_dir, args
        )

        report["subjects"].append(subject_row)
        report["sessions"].extend(session_rows)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    subject_rows = report["subjects"]
    subject_fields = [
        "subject_id",
        "session1_cv_accuracy",
        "session2_cv_accuracy",
        "mean_cv_accuracy",
        "group",
        "trials",
    ]

    for group in ("normal", "bci_blind"):
        with (args.output_dir / f"{group}.csv").open(
            "x", newline="", encoding="utf-8-sig"
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=subject_fields,
            )
            writer.writeheader()
            writer.writerows(
                row for row in subject_rows
                if row["group"] == group
            )
    session_fields = [
        key for key in report["sessions"][0]
        if key != "fold_accuracies"
    ]
    with (args.output_dir / "sessions.csv").open(
        "x", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=session_fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(report["sessions"])
    report["counts"] = {
        group: sum(
            row["group"] == group
            for row in subject_rows
        )
        for group in ("normal", "bci_blind")
    }
    report["status"] = "complete"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Complete (subjects): {report['counts']}; {report_path}",
        flush=True,
    )

if __name__ == "__main__":
    main()
