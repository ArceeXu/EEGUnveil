"""OpenBMI recorded-trial inference and replay protocol (not live acquisition)."""
import json
import math
from pathlib import Path
import socket
import struct
import sys

from torch_bootstrap import prepare_torch_dlls
prepare_torch_dlls()
import joblib
import mne
import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.offline.train_v65_distill import EEGNetLight

DEFAULT_MODEL = ROOT / "models/offline/lambda_analysis/lambda0.5625_seed42"
LABELS = ("左手", "右手")


class BCIProcessor:
    def __init__(self, model_dir=DEFAULT_MODEL):
        folder = Path(model_dir)
        self.config = json.loads((folder / "model_config.json").read_text(encoding="utf-8-sig"))
        cfg = self.config
        if cfg["n_classes"] != 2:
            raise ValueError("请选择 OpenBMI 左右手二分类模型目录。")
        self.channels = cfg["channel_names"]
        self.n_times = cfg["n_times"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EEGNetLight(cfg["n_channels"], self.n_times, n_classes=2,
                                 csp_dim=cfg["csp_dim"]).to(self.device)
        self.model.load_state_dict(torch.load(folder / "best_model.pth",
                                              map_location=self.device, weights_only=True))
        self.model.eval()
        self.csp = joblib.load(folder / "best_csp.pkl")
        self.scalers = joblib.load(folder / "best_scalers.pkl")

    def align(self, trial, channels):
        mapping = {str(name).strip().lower(): i for i, name in enumerate(channels)}
        missing = [name for name in self.channels if name.lower() not in mapping]
        if missing:
            raise ValueError("数据缺少模型通道：" + ", ".join(missing))
        trial = np.asarray(trial, dtype=np.float32)[
            [mapping[name.lower()] for name in self.channels]]
        if trial.shape != (len(self.channels), self.n_times):
            raise ValueError(f"试次尺寸 {trial.shape} 与模型要求不一致。")
        return trial

    def predict(self, trial, channels):
        trial = self.align(trial, channels)
        features = self.csp.transform(trial[None]).astype(np.float32)
        scaled = trial.copy()
        for ch in range(len(self.channels)):
            scaled[ch] = self.scalers[ch].transform(scaled[ch].reshape(1, -1)).ravel()
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(scaled[None]).to(self.device),
                                torch.from_numpy(features).to(self.device))
            probabilities = logits.softmax(dim=1)[0].cpu().numpy()
        prediction = int(probabilities.argmax())
        return prediction, float(probabilities[prediction])


def load_trials(path, phase="test"):
    """Match offline continuous FIR filtering, onset convention and label mapping."""
    key = f"EEG_MI_{phase}"
    source = loadmat(str(path), variable_names=[key], squeeze_me=True,
                     struct_as_record=False)[key]
    fs = int(source.fs)
    channels = np.asarray(source.chan, dtype=str).reshape(-1).tolist()
    signal = np.asarray(source.x, dtype=np.float64).T * 1e-6
    signal = mne.filter.filter_data(signal, fs, 0.5, 100,
                                   method="fir", fir_design="firwin", verbose=False)
    signal = mne.filter.notch_filter(signal, fs, [50],
                                    method="fir", fir_design="firwin", verbose=False)
    onsets = np.asarray(source.t).reshape(-1).astype(int)
    labels = (np.asarray(source.y_dec).reshape(-1).astype(int) == 1).astype(int)
    divisor = math.gcd(fs, 250)
    for index, (onset, label) in enumerate(zip(onsets, labels), 1):
        epoch = signal[:, onset:onset + 4 * fs]
        if epoch.shape[1] != 4 * fs:
            raise ValueError(f"第 {index} 个试次不足 4 秒。")
        trial = resample_poly(epoch, 250 // divisor, fs // divisor, axis=-1).astype(np.float32)
        yield index, len(onsets), trial, int(label), channels


def receive_exact(connection, size):
    result = bytearray()
    while len(result) < size:
        block = connection.recv(size - len(result))
        if not block:
            if not result:
                return None
            raise ConnectionError("网络数据未接收完整，发送端已断开。")
        result.extend(block)
    return bytes(result)


def send_trial(connection, item):
    index, total, trial, label, channels = item
    trial = np.asarray(trial, dtype="<f4")
    header = json.dumps(dict(protocol="openbmi_trial_v1", index=index, total=total,
                             shape=trial.shape, label=label, channels=channels)).encode()
    connection.sendall(struct.pack("!I", len(header)) + header + trial.tobytes())


def receive_trial(connection):
    prefix = receive_exact(connection, 4)
    if prefix is None:
        return None
    length = struct.unpack("!I", prefix)[0]
    if not 0 < length <= 1048576:
        raise ValueError("发送端协议不匹配，请使用本项目新版 sender.py。")
    header = json.loads(receive_exact(connection, length))
    n_channels, n_times = header["shape"]
    if header["protocol"] != "openbmi_trial_v1" or not (1 <= n_channels <= 128 and n_times == 1000):
        raise ValueError("接收到的不是 OpenBMI 4 秒试次。")
    payload = receive_exact(connection, n_channels * n_times * 4)
    if payload is None:
        raise ConnectionError("发送端在传输试次前断开。")
    trial = np.frombuffer(payload, dtype="<f4").reshape(n_channels, n_times).copy()
    return header["index"], header["total"], trial, header["label"], header["channels"]


if __name__ == "__main__":
    print("请运行 python src/online/main_app.py 打开 OpenBMI 图形界面。")

