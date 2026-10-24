"""Split OpenBMI MI data separately for each subject/session (see docs/openbmi_split.md)."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
from math import gcd


ROOT = Path(__file__).resolve().parents[2]
PATTERN = re.compile(r"sess(\d+)_subj(\d+)_EEG_MI\.mat$")
RUNS = ("EEG_MI_train", "EEG_MI_test")


def integer_vector(value, name):
    values = np.asarray(value).reshape(-1)
    return values.astype(np.int64)


def read_run(path, run, tmin=0.0, tmax=4.0, sfreq=None):
    """读取一个 MAT 文件中的一个阶段，并提取各 trial。
    run：要读取的阶段名称，即 EEG_MI_train 或 EEG_MI_test。
    tmin, tmax：相对于 trial onset 的时间窗口（秒）。
    sfreq：目标采样率（Hz）；如果为 None，则保持原始采样率。
    """
    mat = loadmat(str(path), variable_names=[run], squeeze_me=True,
                  struct_as_record=False)
    data = mat[run]
    fs = float(data.fs)
    fs = int(fs)
    labels = integer_vector(data.y_dec, "y_dec")
    events = integer_vector(data.t, "t")
    channels = np.asarray(data.chan, dtype=str).reshape(-1)
    signal = np.asarray(data.x)
    mapping = {int(row[0]): str(row[1]).strip().lower()
               for row in np.asarray(getattr(data, "class"))}
    a, b = round(tmin * fs), round(tmax * fs)
    starts, stops = events + a, events + b
    X = np.stack([signal[start:stop].T for start, stop in zip(starts, stops)])
    X = X.astype(np.float32)
    X *= 1e-6  
    target_fs = fs if sfreq is None else sfreq
    if target_fs != fs:
        divisor = gcd(fs, target_fs)
        # Each trial is resampled independently, with anti-alias filtering.
        X = resample_poly(X, target_fs // divisor, fs // divisor, axis=-1)
    return {
        "X": X.astype(np.float32),
        "y": (labels == 1).astype(np.int64),  # 0=left, 1=right
        "trial_id": np.arange(len(labels), dtype=np.int64),
        "onset_sample": events,
        "source_run": np.full(len(labels), run),
        "channel_names": channels,
        "sfreq": target_fs,
        "original_sfreq": fs,
    }


def stratified_indices(y, test_size, seed):
    """Split complete trials; per-class rounding may slightly alter test_size."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == label))
        count = int(np.floor(len(indices) * test_size + 0.5))
        test.extend(indices[:count])
        train.extend(indices[count:])
    return np.sort(train), np.sort(test)


TRIAL_KEYS = ("X", "y", "trial_id", "onset_sample", "source_run")


def save_split(path, data, indices, subject, session, config):
    payload = {key: data[key][indices] for key in TRIAL_KEYS}
    payload.update({key: data[key] for key in
                    ("channel_names", "sfreq", "original_sfreq")})
    payload.update(subject_id=np.full(len(indices), subject, dtype=np.int64),
                   session_id=np.full(len(indices), session, dtype=np.int64),
                   class_names=np.array(["left", "right"]), unit="V",
                   metadata_json=json.dumps(config, ensure_ascii=False))
    # Exclusive creation prevents accidental overwrites of frozen test splits.
    with path.open("xb") as stream:
        np.savez_compressed(stream, **payload)
    return {"trials": len(indices), "shape": list(payload["X"].shape),
            "left": int(np.sum(payload["y"] == 0)),
            "right": int(np.sum(payload["y"] == 1))}


def split_file(path, output, subject, session, args):
    config = {"source_file": str(path.resolve()), "subject": subject,
              "session": session, "mode": args.mode,
              "test_size": args.test_size if args.mode == "random" else None,
              "seed": args.seed, "tmin": args.tmin, "tmax": args.tmax,
              "requested_sfreq": args.sfreq, "event_index_convention": "direct_offset",
              "label_mapping": {"0": "left", "1": "right"},
              "unit": "V", "normalization": "none", "bandpass": "none"}
    folder = output / f"sub-{subject:02d}" / f"ses-{session:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    if any((folder / name).exists() for name in ("train.npz", "test.npz", "metadata.json")):
        raise FileExistsError(f"Output already exists: {folder}; use a new --output-dir")
    blocks = [read_run(path, run, args.tmin, args.tmax, args.sfreq) for run in RUNS]
    if args.mode == "official":
        pairs = [(block, np.arange(len(block["y"]))) for block in blocks]
    else:
        combined = dict(blocks[0])
        combined.update({key: np.concatenate([block[key] for block in blocks])
                         for key in TRIAL_KEYS})
        # Stable per-session seed: selecting other files cannot change this split.
        seed = np.random.SeedSequence([args.seed, subject, session])
        train, test = stratified_indices(combined["y"], args.test_size, seed)
        pairs = [(combined, train), (combined, test)]
    stats = {}
    for name, (data, indices) in zip(("train", "test"), pairs):
        stats[name] = save_split(folder / f"{name}.npz", data, indices,
                                 subject, session, config)
    config.update(stats)
    (folder / "metadata.json").write_text(json.dumps(config, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Data/OpenBMI/raw")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=("official", "random"), default="official",
                        help="official: preserve source runs; random: pool both runs then stratify")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Used only in random mode (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subjects", nargs="+", type=int)
    parser.add_argument("--sessions", nargs="+", type=int, choices=(1, 2), default=[1, 2])
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--sfreq", type=int, help="Optional resampling Hz; default keeps native rate")
    args = parser.parse_args()
    files = {}
    for path in sorted(args.data_dir.rglob("sess*_subj*_EEG_MI.mat")):
        match = PATTERN.fullmatch(path.name)
        if not match:
            continue
        session, subject = map(int, match.groups())
        if session not in args.sessions or (args.subjects and subject not in args.subjects):
            continue
        key = (subject, session)
        files[key] = path
    if not files:
        parser.error(f"No matching MI files under {args.data_dir}")
    subjects = sorted(set(args.subjects or [s for s, _ in files]))
    missing = [(s, session) for s in subjects for session in args.sessions
               if (s, session) not in files]
    if missing:
        parser.error(f"Missing requested subject/session pairs: {missing}")
    output = args.output_dir or ROOT / "Data/OpenBMI/splits" / args.mode
    print(f"Found {len(subjects)} subjects, {len(files)} sessions. Output: {output}", flush=True)
    for (subject, session), path in sorted(files.items()):
        print(f"Processing subject={subject:02d}, session={session:02d} ...", flush=True)
        report = split_file(path, output, subject, session, args)
        print(f"  train={report['train']}, test={report['test']}", flush=True)


if __name__ == "__main__":
    main()
