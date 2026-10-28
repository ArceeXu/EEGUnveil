#!/usr/bin/env bash
set -euo pipefail

cd /openbmi_dataset/bci_project

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

COMMON=(
  --stage train
  --groups-dir /openbmi_dataset/Data/OpenBMI/groups/csp70_20
  --data-root /openbmi_dataset/Data
  --split-file outputs/splits/openbmi_subject_holdout.json
  --split-seed 2026
  --batch-size 32
  --epochs 120
  --num-workers 0
)

mkdir -p outputs/logs/holdout

for seed in 42 2024 3407; do
  for method in A B C D; do
    OUT="models/offline/holdout/${method}_seed${seed}"

    if [[ -s "$OUT/validation_metrics.json" \
       && -s "$OUT/subject_split.json" \
       && -s "$OUT/model_config.json" \
       && -s "$OUT/best_model.pth" \
       && -s "$OUT/best_csp.pkl" \
       && -s "$OUT/best_scalers.pkl" ]]; then
      echo "跳过已完成训练：$OUT"
      continue
    fi

    if [[ -e "$OUT" ]]; then
      echo "发现不完整实验目录：$OUT"
      echo "请先检查日志并保留失败目录，再单独重跑该实验。"
      exit 1
    fi

    case "$method" in
      A) EXTRA=(--sampling natural) ;;
      B) EXTRA=(--sampling mixed --mix-lambda 0.75 --no-distillation) ;;
      C) EXTRA=(--sampling mixed --mix-lambda 0.75) ;;
      D) EXTRA=(--sampling natural --no-distillation) ;;
    esac

    python -u src/offline/run_openbmi_holdout.py \
      "${COMMON[@]}" \
      "${EXTRA[@]}" \
      --seed "$seed" \
      --output-dir "$OUT" \
      2>&1 | tee "outputs/logs/holdout/${method}_seed${seed}.log"
  done
done
