#!/usr/bin/env bash
set -euo pipefail
cd /openbmi_dataset/bci_project

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

mkdir -p outputs/logs/ready_tests
exec 9>outputs/ready_tests.lock
flock -n 9 || {
  echo "已有测试调度器正在运行，请勿重复启动。"
  exit 1
}

test_one() {
  local out="$1"
  local tag="$2"
  local file

  # 训练必须完整结束后才允许测试。
  for file in best_model.pth best_csp.pkl best_scalers.pkl \
              model_config.json validation_metrics.json subject_split.json; do
    if [[ ! -s "$out/$file" ]]; then
      echo "[跳过，训练未完成或不存在] $out"
      return 0
    fi
  done

  if [[ -s "$out/test_metrics.json" \
     && -s "$out/test_predictions.csv" \
     && -s "$out/subject_metrics.csv" ]]; then
    echo "[跳过，已有完整测试结果] $out"
    return 0
  fi

  # 不覆盖部分生成的测试结果，出现这种情况先人工检查。
  for file in test_metrics.json test_predictions.csv subject_metrics.csv; do
    if [[ -e "$out/$file" ]]; then
      echo "[停止] 发现不完整测试结果：$out/$file"
      echo "请保留文件并检查之前的测试日志。"
      return 1
    fi
  done

  echo "[开始测试] $tag $(date -Iseconds)"

  python -u src/offline/run_openbmi_holdout.py \
    --stage test \
    --groups-dir /openbmi_dataset/Data/OpenBMI/groups/csp70_20 \
    --data-root /openbmi_dataset/Data \
    --split-file outputs/splits/openbmi_subject_holdout.json \
    --split-seed 2026 \
    --batch-size 32 \
    --num-workers 0 \
    --output-dir "$out" \
    2>&1 | tee "outputs/logs/ready_tests/${tag}.log"

  echo "[测试完成] $tag"
}

for seed in 42 2024 3407; do
  test_one \
    "models/offline/holdout/B_seed${seed}" \
    "B_lambda075_seed${seed}"

  test_one \
    "models/offline/holdout/C_seed${seed}" \
    "C_lambda075_seed${seed}"

  test_one \
    "models/offline/lambda_analysis/lambda0.5625_seed${seed}" \
    "M1_lambda05625_seed${seed}"

  test_one \
    "models/offline/normal_only/N0_seed${seed}" \
    "N0_seed${seed}"

  test_one \
    "models/offline/normal_only/N1_seed${seed}" \
    "N1_seed${seed}"
done

echo "本轮检查完成：所有当时已就绪的模型均已测试，未完成训练的模型已跳过。"
