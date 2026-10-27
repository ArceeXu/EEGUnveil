import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--all",
    action="store_true",
    help="下载全部54人；默认只下载第1人",
)
args = parser.parse_args()

# 从脚本位置定位项目，不依赖终端当前目录
project_root = Path(__file__).resolve().parents[2]
save_dir = project_root / "Data" / "OpenBMI" / "raw"
save_dir.mkdir(parents=True, exist_ok=True)

# 后续加载也应使用同一个数据路径
os.environ["MNE_DATA"] = str(save_dir)
os.environ["MNE_DATASETS_LEE2019-MI_PATH"] = str(save_dir)

from moabb.datasets import Lee2019_MI
from scipy.io import whosmat

subjects = range(1, 54) if args.all else [1]
failures = []
success_count = 0

for subject in subjects:
    for session in (1, 2):
        print(
            f"\n正在处理：受试者 {subject:02d}，"
            f"Session {session}",
            flush=True,
        )

        try:
            dataset = Lee2019_MI(
                sessions=[session],
                train_run=True,
                test_run=False,
            )

            paths = dataset.data_path(
                subject=subject,
                path=str(save_dir),
                force_update=False,
            )

            for item in paths:
                path = Path(item)

                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"文件不存在或为空：{path}")

                # 检查文件结构，不将全部脑电加载进内存
                fields = {name for name, _, _ in whosmat(str(path))}
                if "EEG_MI_train" not in fields:
                    raise RuntimeError(
                        f"缺少 EEG_MI_train 字段：{path}"
                    )

                size_mb = path.stat().st_size / (1024 ** 2)
                print(f"文件：{path}")
                print(f"大小：{size_mb:.1f} MB")
                print("MAT文件结构检查通过")
                success_count += 1

        except Exception as exc:
            failures.append((subject, session, str(exc)))
            print(f"失败：{exc}", flush=True)

print(f"\n完成：{success_count} 个文件通过结构检查。")

if failures:
    print("\n以下项目未完成：")
    for subject, session, error in failures:
        print(f"subject={subject}, session={session}: {error}")
    raise SystemExit(1)

print(f"保存目录：{save_dir}")