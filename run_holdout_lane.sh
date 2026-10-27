#!/usr/bin/env bash
set -euo pipefail

cd /openbmi_dataset/bci_project

LANE="${1:?需要指定 lane1 或 lane2}"

case "$LANE" in
  lane1) SEEDS=(42 3407) ;;
  lane2) SEEDS=(2024) ;;
  *) echo "仅支持 lane1 或 lane2"; exit 1 ;;
esac

mkdir -p outputs/logs/holdout

# 防止同一路线被重复启动。
exec 9>"outputs/${LANE}.lock"
flock -n 9 || {
  echo "$LANE 已经在运行，退出。"
  exit 1
}

# 原串行调度器若仍存在，必须保持暂停状态。
check_old_scheduler() {
  local args state
  args="$(ps -p 6386 -o args= 2>/dev/null || true)"
  if [[ "$args" == "bash run_four_holdout.sh" ]]; then
    state="$(ps -p 6386 -o stat=)"
    if [[ "$state" != *T* ]]; then
      echo "原调度器 6386 未暂停，为避免重复任务，退出。"
      exit 1
    fi
  fi
}

check_old_scheduler

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

if [[ "$LANE" == "lane1" ]]; then
  echo "等待现有 C_seed42 训练进程 19908 自然结束……"
  while ps -p 19908 -o args= 2>/dev/null \
    | grep -q '[r]un_openbmi_holdout.py'; do
    sleep 15
  done
  echo "现有训练进程已退出，将检查结果是否完整。"
fi

for seed in "${SEEDS[@]}"; do
  for method in A B C D; do
    check_old_scheduler

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
      echo "发现不完整目录：$OUT"
      echo "本路线停止，请检查该实验日志；不会覆盖或删除文件。"
      exit 1
    fi

    case "$method" in
      A) EXTRA=(--sampling natural) ;;
      B) EXTRA=(--sampling mixed --mix-lambda 0.75 --no-distillation) ;;
      C) EXTRA=(--sampling mixed --mix-lambda 0.75) ;;
      D) EXTRA=(--sampling natural --no-distillation) ;;
    esac

    echo "[$LANE] 开始 ${method}_seed${seed}：$(date -Is)"

    python -u src/offline/run_openbmi_holdout.py \
      "${COMMON[@]}" \
      "${EXTRA[@]}" \
      --seed "$seed" \
      --output-dir "$OUT" \
      2>&1 | tee "outputs/logs/holdout/${method}_seed${seed}.log"

    echo "[$LANE] 完成 ${method}_seed${seed}：$(date -Is)"
  done
done

echo "[$LANE] 全部完成"
