#!/usr/bin/env bash
set -euo pipefail
cd /openbmi_dataset/bci_project

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

mkdir -p outputs/logs/m0_05625
exec 9>outputs/m0_05625.lock
flock -n 9 || {
  echo "该调度器已在运行，请勿重复启动。"
  exit 1
}

[[ -s outputs/splits/openbmi_subject_holdout.json ]] || {
  echo "缺少原受试者划分，停止。"
  exit 1
}

COMMON=(
  --groups-dir /openbmi_dataset/Data/OpenBMI/groups/csp70_20
  --data-root /openbmi_dataset/Data
  --split-file outputs/splits/openbmi_subject_holdout.json
  --split-seed 2026
  --batch-size 32
  --num-workers 0
)

run_one() {
  local seed="$1"
  local out="models/offline/selected_lambda_no_kd/M0_lambda05625_seed${seed}"
  local file complete=1

  for file in best_model.pth best_csp.pkl best_scalers.pkl \
              model_config.json validation_metrics.json subject_split.json; do
    if [[ ! -s "$out/$file" ]]; then
      complete=0
    fi
  done

  if [[ "$complete" -eq 0 ]]; then
    if [[ -e "$out" ]]; then
      echo "发现不完整训练目录，请检查后处理：$out"
      return 1
    fi

    echo "[开始训练] seed=$seed $(date -Iseconds)"

    python -u src/offline/run_openbmi_holdout.py \
      --stage train \
      "${COMMON[@]}" \
      --sampling mixed \
      --mix-lambda 0.5625 \
      --no-distillation \
      --seed "$seed" \
      --epochs 120 \
      --output-dir "$out" \
      2>&1 | tee "outputs/logs/m0_05625/train_seed${seed}.log"
  else
    echo "[跳过完整训练] $out"
  fi

  # 确认训练产物齐全后再测试。
  for file in best_model.pth best_csp.pkl best_scalers.pkl \
              model_config.json validation_metrics.json subject_split.json; do
    if [[ ! -s "$out/$file" ]]; then
      echo "训练产物缺失：$out/$file"
      return 1
    fi
  done

  if [[ -s "$out/test_metrics.json" \
     && -s "$out/test_predictions.csv" \
     && -s "$out/subject_metrics.csv" ]]; then
    echo "[跳过完整测试] $out"
    return 0
  fi

  for file in test_metrics.json test_predictions.csv subject_metrics.csv; do
    if [[ -e "$out/$file" ]]; then
      echo "发现不完整测试文件，请检查后处理：$out/$file"
      return 1
    fi
  done

  echo "[开始测试] seed=$seed $(date -Iseconds)"

  python -u src/offline/run_openbmi_holdout.py \
    --stage test \
    "${COMMON[@]}" \
    --output-dir "$out" \
    2>&1 | tee "outputs/logs/m0_05625/test_seed${seed}.log"

  echo "[训练和测试完成] seed=$seed"
}

lane1() {
  run_one 42
}

lane2() {
  run_one 2024
  run_one 3407
}

lane1 >outputs/logs/m0_05625/lane1.log 2>&1 &
pid1=$!

lane2 >outputs/logs/m0_05625/lane2.log 2>&1 &
pid2=$!

status=0
wait "$pid1" || status=1
wait "$pid2" || status=1

if [[ "$status" -eq 0 ]]; then
  echo "三个模型的训练和测试全部完成。"
else
  echo "有任务失败，请检查 lane1.log 和 lane2.log。"
fi
exit "$status"
