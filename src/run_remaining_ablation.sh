#!/bin/bash
# run_remaining_ablation.sh — resume ablation from where we left off
# Skips already-completed (base, full_einn@25) using --skip-base
set -e
cd "$(dirname "$0")"
mkdir -p logs

SCIBERT="models/scibert_safe"
SEED=42

# Remaining: 25% missing EPC variants (base+full_einn done)
echo "===== 25%: EPC variants ====="
for variant in shuffled_summary random_summary residual_only no_gate; do
  logfile="logs/ablation_${variant}_25pct.log"
  echo "[$(date)] Starting: variant=$variant fraction=0.25 -> $logfile"
  python ablation_epc_evidence_control.py \
    --variant "$variant" \
    --train-fraction 0.25 \
    --skip-base \
    --scibert-dir "$SCIBERT" \
    --seed "$SEED" \
    --out-dir results/ablation \
    --fig-dir figures \
    > "$logfile" 2>&1
  echo "[$(date)] Finished: variant=$variant fraction=0.25"
done

# 50%, 75%, 100%: all EPC variants (base already done)
for frac in 0.50 0.75 1.00; do
  frac_pct=$(python -c "print(int(round($frac * 100)))")
  echo "===== ${frac_pct}%: EPC variants ====="
  for variant in full_einn shuffled_summary random_summary residual_only no_gate; do
    logfile="logs/ablation_${variant}_${frac_pct}pct.log"
    echo "[$(date)] Starting: variant=$variant fraction=$frac -> $logfile"
    python ablation_epc_evidence_control.py \
      --variant "$variant" \
      --train-fraction "$frac" \
      --skip-base \
      --scibert-dir "$SCIBERT" \
      --seed "$SEED" \
      --out-dir results/ablation \
      --fig-dir figures \
      > "$logfile" 2>&1
    echo "[$(date)] Finished: variant=$variant fraction=$frac"
  done
done

echo "====== All EPC runs done ======"

# Generate final plots and tables
echo "[$(date)] Generating final plots and LaTeX table..."
python ablation_epc_evidence_control.py \
  --only-plot \
  --scibert-dir "$SCIBERT" \
  --out-dir results/ablation \
  --fig-dir figures \
  >> logs/final_plots.log 2>&1

echo "[$(date)] Results in results/ablation/"
echo "[$(date)] Figures in figures/"
