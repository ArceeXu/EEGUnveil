"""Compatibility filename: OpenBMI MAT replay evaluation, not live-stream evaluation."""
import argparse
import csv
from pathlib import Path
from receiver import BCIProcessor, DEFAULT_MODEL, LABELS, load_trials


def main():
    parser = argparse.ArgumentParser(description="OpenBMI 单个 MAT 记录回放与识别")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    parser.add_argument("--phase", choices=["train", "test"], default="test")
    parser.add_argument("--output", help="可选：保存逐试次 CSV 的路径")
    args = parser.parse_args()
    processor = BCIProcessor(args.model_dir)
    rows = []
    print("正在读取并预处理记录……", flush=True)
    for index, total, trial, truth, channels in load_trials(args.data, args.phase):
        prediction, confidence = processor.predict(trial, channels)
        rows.append(dict(trial=index, truth=truth, prediction=prediction, confidence=confidence))
        print(f"{index}/{total} 真实：{LABELS[truth]} 预测：{LABELS[prediction]} "
              f"置信度：{confidence:.1%}", flush=True)
    if rows:
        accuracy = sum(r["truth"] == r["prediction"] for r in rows) / len(rows)
        print(f"当前记录回放符合率：{accuracy:.2%}。这不是完整留出受试者评估。")
    if args.output and rows:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()

