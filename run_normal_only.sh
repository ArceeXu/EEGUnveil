#!/usr/bin/env bash
# Two lanes: N0 and N1, each runs three seeds sequentially.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p outputs/logs/normal_only
exec 9>outputs/normal_only.lock
flock -n 9 || { echo "仅正常组调度器已在运行"; exit 1; }
[[ -s outputs/splits/openbmi_subject_holdout.json ]] || { echo "缺少原受试者划分"; exit 1; }

run_lane() {
  local method="$1" seed out file complete
  local -a extra=()
  if [[ "$method" == N0 ]]; then extra=(--no-distillation); fi
  for seed in 42 2024 3407; do
    out="models/offline/normal_only/${method}_seed${seed}"
    complete=1
    for file in best_model.pth best_csp.pkl best_scalers.pkl model_config.json validation_metrics.json subject_split.json; do
      [[ -s "$out/$file" ]] || complete=0
    done
    if [[ "$complete" == 1 ]]; then
      echo "跳过已完成：$out"
      continue
    fi
    if [[ -e "$out" ]]; then
      echo "发现不完整目录，停止本路，请检查：$out"
      return 1
    fi
    echo "开始 $method seed=$seed $(date -Iseconds)"
    python -u src/offline/run_openbmi_holdout.py \
      --stage train --train-population normal --sampling natural \
      --groups-dir /openbmi_dataset/Data/OpenBMI/groups/csp70_20 \
      --data-root /openbmi_dataset/Data \
      --split-file outputs/splits/openbmi_subject_holdout.json \
      --split-seed 2026 --seed "$seed" --batch-size 32 \
      --epochs 120 --num-workers 0 "${extra[@]}" --output-dir "$out" \
      2>&1 | tee "outputs/logs/normal_only/${method}_seed${seed}.log"
    echo "完成 $method seed=$seed"
  done
}
run_lane N0 >outputs/logs/normal_only/N0_lane.log 2>&1 &
pid0=$!
run_lane N1 >outputs/logs/normal_only/N1_lane.log 2>&1 &
pid1=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if [[ "$status" == 0 ]]; then echo "仅正常组六次训练全部完成"; else echo "有训练失败，请检查两路日志"; fi
exit "$status"
