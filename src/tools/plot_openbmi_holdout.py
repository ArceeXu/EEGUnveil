import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--font", type=Path, required=True)
p.add_argument("--output-dir", type=Path, required=True)
args = p.parse_args()

if not args.font.is_file():
    raise FileNotFoundError(f"找不到宋体字体文件：{args.font}")

font_manager.fontManager.addfont(str(args.font))
font_name = font_manager.FontProperties(fname=str(args.font)).get_name()
plt.rcParams.update({
    "font.family": font_name,
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.titlesize": 14,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})

PALETTE = [
    "#588896", "#63b8ac", "#8ed2c7",
    "#b0e4db", "#72a8b5", "#d0f0e7",
]
LABELS = {
    "A": "自然比例＋蒸馏",
    "B": "比例混合＋无蒸馏",
    "C": "比例混合＋蒸馏",
    "D": "自然比例＋无蒸馏",
}

args.output_dir.mkdir(parents=True, exist_ok=True)
records, subject_tables, splits = [], [], []

for path in sorted(args.root.glob("*_seed*/test_metrics.json")):
    method = path.parent.name.split("_")[0]
    if method not in LABELS:
        continue
    m = json.loads(path.read_text(encoding="utf-8"))
    records.append({
        "method": method,
        "seed": m["seed"],
        "总体": m["subject_macro_accuracy"] * 100,
        "正常组": m["group_subject_macro_accuracy"]["normal"] * 100,
        "BCI盲组": m["group_subject_macro_accuracy"]["bci_blind"] * 100,
    })
    table = pd.read_csv(path.parent / "subject_metrics.csv")
    table["method"], table["seed"] = method, m["seed"]
    subject_tables.append(table)
    splits.append(json.loads(
        (path.parent / "subject_split.json").read_text(encoding="utf-8")
    ))

df = pd.DataFrame(records)
if df.empty:
    raise ValueError("没有找到测试结果")
if set(df["method"]) != set(LABELS):
    raise ValueError("需要完整的 A、B、C、D 四组结果")
if df.duplicated(["method", "seed"]).any():
    raise ValueError("方法与 seed 重复")
if any(s != splits[0] for s in splits):
    raise ValueError("不同实验使用了不同受试者划分")

seed_sets = [
    set(df.loc[df.method == method, "seed"]) for method in LABELS
]
if any(s != seed_sets[0] for s in seed_sets):
    raise ValueError("四组实验的 seed 不一致")

df.to_csv(args.output_dir / "summary.csv", index=False, encoding="utf-8-sig")
n_seeds = len(seed_sets[0])


def save(fig, name):
    fig.savefig(args.output_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(args.output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


categories = ["总体", "正常组", "BCI盲组"]
x = np.arange(len(categories))
width = 0.19
fig, ax = plt.subplots(figsize=(10, 5))

for i, method in enumerate(LABELS):
    values = df[df.method == method][categories]
    means = values.mean().to_numpy()
    errors = values.std(ddof=1).to_numpy() if n_seeds > 1 else None
    ax.bar(
        x + (i - 1.5) * width, means, width,
        yerr=errors, capsize=3,
        label=LABELS[method], color=PALETTE[i],
        edgecolor="#48636b", linewidth=0.5,
    )

ax.set_xticks(x, categories)
ax.set_ylabel("受试者平均准确率（%）")
ax.set_ylim(0, 100)
ax.set_title("四组方法在留出受试者上的表现")
ax.legend(frameon=False, ncol=2)
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)
fig.text(
    0.5, 0.01,
    f"共 {n_seeds} 个训练随机种子"
    + ("；误差线为种子间标准差" if n_seeds > 1 else "；单种子不绘制误差线"),
    ha="center",
)
fig.tight_layout(rect=(0, 0.05, 1, 1))
save(fig, "四组方法对比")

subjects = pd.concat(subject_tables, ignore_index=True)
if subjects.duplicated(["subject_id", "seed", "method"]).any():
    raise ValueError("逐受试者结果存在重复记录")
expected_subjects = set(splits[0]["test"])
for (method, seed), table in subjects.groupby(["method", "seed"]):
    if set(table.subject_id) != expected_subjects:
        raise ValueError(f"{method}, seed={seed} 的测试受试者不完整")
    if not table.accuracy.between(0, 1).all():
        raise ValueError("准确率必须位于 [0, 1]，且不能包含缺失值")
blind = subjects[subjects.group == "bci_blind"]
pivot = blind.pivot(
    index=["subject_id", "seed"], columns="method", values="accuracy"
)
if pivot.empty or pivot.isna().any().any():
    raise ValueError("BCI 盲组配对结果不完整，不能计算改善")
delta = pd.DataFrame({
    "混合策略增益": 100 * (pivot["C"] - pivot["A"]),
    "蒸馏增益": 100 * (pivot["C"] - pivot["B"]),
})
delta = delta.groupby(level="subject_id").mean().sort_index()
delta.to_csv(
    args.output_dir / "blind_subject_improvement.csv",
    encoding="utf-8-sig",
)

fig, ax = plt.subplots(figsize=(max(8, len(delta) * 0.8), 5))
x = np.arange(len(delta))
ax.bar(x - 0.18, delta.iloc[:, 0], 0.36,
       color=PALETTE[0], label="完整方法减自然比例蒸馏")
ax.bar(x + 0.18, delta.iloc[:, 1], 0.36,
       color=PALETTE[1], label="完整方法减比例混合无蒸馏")
ax.axhline(0, color="#555555", linewidth=0.8)
ax.set_xticks(x, [f"受试者{sid:02d}" for sid in delta.index])
ax.set_ylabel("准确率变化（百分点）")
ax.set_title("BCI盲组留出受试者的个体改善")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)
fig.tight_layout()
save(fig, "BCI盲组逐受试者改善")
