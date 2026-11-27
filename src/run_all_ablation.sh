#!/bin/bash
# run_all_ablation.sh — launch all 6 ablation variants × 4 fractions
# Usage: bash run_all_ablation.sh

set -e
cd "$(dirname "$0")"
mkdir -p logs

VARIANTS=("base" "full_einn" "shuffled_summary" "random_summary" "residual_only" "no_gate")
FRACTIONS=("0.25" "0.50" "0.75" "1.00")
SCIBERT="models/scibert_safe"
SEED=42

for variant in "${VARIANTS[@]}"; do
  for frac in "${FRACTIONS[@]}"; do
    frac_pct=$(python -c "print(int(round($frac * 100)))")
    logfile="logs/ablation_${variant}_${frac_pct}pct.log"
    echo "[$(date)] Starting: variant=$variant fraction=$frac -> $logfile"
    python ablation_epc_evidence_control.py \
      --variant "$variant" \
      --train-fraction "$frac" \
      --scibert-dir "$SCIBERT" \
      --seed "$SEED" \
      --out-dir results/ablation \
      --fig-dir figures \
      > "$logfile" 2>&1
    echo "[$(date)] Finished: variant=$variant fraction=$frac (exit code $?)"
  done
done

echo "====== All Done ======"

# Generate final plots and tables from the accumulated CSV files
echo "[$(date)] Generating final plots and LaTeX table..."
python ablation_epc_evidence_control.py \
  --only-plot \
  --scibert-dir "$SCIBERT" \
  --out-dir results/ablation \
  --fig-dir figures \
  >> logs/final_plots.log 2>&1

echo "[$(date)] Results in results/ablation/"
echo "[$(date)] Figures in figures/"
