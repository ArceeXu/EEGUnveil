"""Replot the fixed lambda=0.5625 comparison from exported subject results."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--font', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    required = {'subject_id', 'group', 'accuracy', 'method', 'seed'}
    if not required.issubset(df.columns):
        raise ValueError(f'缺少字段：{required - set(df.columns)}')
    blind = df[df.group == 'bci_blind'].copy()
    if blind.duplicated(['subject_id', 'seed', 'method']).any():
        raise ValueError('存在重复受试者结果')
    if not blind.accuracy.between(0, 1).all():
        raise ValueError('准确率必须是0至1之间的数值')
    expected_ids = {7, 12, 17, 24, 30, 51, 54}
    expected_seeds = {42, 2024, 3407}
    if set(blind.method) != {'N0', 'N1', 'M0', 'M1'}:
        raise ValueError('需要N0、N1、M0、M1四组结果')
    for method in ('N0', 'N1', 'M0', 'M1'):
        table = blind[blind.method == method]
        if set(table.seed) != expected_seeds:
            raise ValueError(f'{method} 的三个种子不完整')
        for seed, rows in table.groupby('seed'):
            if set(rows.subject_id) != expected_ids:
                raise ValueError(f'{method}/{seed} 的7名BCI盲受试者不完整')
    font_manager.fontManager.addfont(str(args.font))
    plt.rcParams.update({
        'font.family': font_manager.FontProperties(fname=str(args.font)).get_name(),
        'font.size': 12, 'axes.unicode_minus': False, 'pdf.fonttype': 42,
        'savefig.dpi': 300,
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores = blind.groupby(['seed', 'method']).accuracy.mean().unstack() * 100
    scores.to_csv(args.output_dir / 'BCI盲组各种子准确率.csv', encoding='utf-8-sig')
    fig, ax = plt.subplots(figsize=(9, 6))
    width = .28
    markers = ['o', 's', '^']
    gains = []
    for i, (normal, mixed) in enumerate((('N0', 'M0'), ('N1', 'M1'))):
        heights = []
        for j, method in enumerate((normal, mixed)):
            values = scores[method].to_numpy()
            mean, std = values.mean(), values.std(ddof=1)
            center = i + (j-.5)*width
            ax.bar(center, mean, width*.92, yerr=std, capsize=5,
                   color=('#588896', '#63b8ac')[j], edgecolor='none', linewidth=0,
                   label=('仅正常组训练', '混合训练（λ=0.5625）')[j] if i == 0 else None)
            for k, value in enumerate(values):
                ax.scatter(center+(k-1)*.045, value, marker=markers[k],
                           s=32, color='#29474f', zorder=4)
            ax.text(center, mean+std+1.5, f'{mean:.2f}%', ha='center', fontsize=11)
            heights.append(max(mean+std, values.max()))
        gain = (scores[mixed]-scores[normal]).mean()
        gains.append({'条件': ('无蒸馏', '有蒸馏')[i], '比较':f'{mixed}-{normal}',
                      '平均提升_百分点':gain})
        y = max(heights)+8
        ax.plot([i-width/2, i-width/2, i+width/2, i+width/2],
                [y-1, y, y, y-1], color='#48636b', linewidth=1)
        ax.text(i, y+1, f'混合训练差值：{gain:+.2f} 个百分点', ha='center', fontsize=11)
    ax.set(xticks=[0,1], xticklabels=['无蒸馏', '有蒸馏'], ylim=(0,100),
           ylabel='BCI盲测试受试者平均准确率（%）',
           title='混合训练对未见BCI盲受试者识别性能的影响')
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.1), ncol=2, frameon=False)
    ax.text(.5, -.20, '仅统计7名留出BCI盲受试者；点表示三个训练种子，误差线为种子间标准差\n'
             '每个种子先对受试者等权平均；差值为描述性结果，不表示统计显著性',
             ha='center', va='top', fontsize=10, transform=ax.transAxes)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(args.output_dir / f'BCI盲组混合训练主对比.{ext}', bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(gains).to_csv(args.output_dir / 'BCI盲组混合收益.csv', index=False, encoding='utf-8-sig')
    print(pd.DataFrame(gains).to_string(index=False))
    print(f'已保存到：{args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
