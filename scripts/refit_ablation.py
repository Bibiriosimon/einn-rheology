"""Rerun each Table 4 variant against its archived split and fixed-base predictions."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("base", "full_einn", "shuffled_summary", "random_summary", "residual_only", "no_gate")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scibert", type=Path, required=True)
    p.add_argument("--knowledge", type=Path, required=True)
    p.add_argument("--base-checkpoints", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    p.add_argument("--fractions", nargs="+", type=int, default=[25, 50, 75, 100], choices=[25, 50, 75, 100])
    p.add_argument("--prior-epochs", type=int, default=40)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    for variant in args.variants:
        for fraction in args.fractions:
            label = f"{fraction}pct"
            run = (args.out / f"{variant}-{label}").resolve()
            data = run / label
            data.mkdir(parents=True)
            for name in ("split_indices.csv", "transrheo_y_predictions_all_splits.csv"):
                shutil.copyfile(ROOT / "runs/ablation" / label / name, data / name)
            checkpoint = (args.base_checkpoints / label / "transrheo_base_ckpt.pt").resolve(strict=True)
            (data / "transrheo_base_ckpt.pt").symlink_to(checkpoint)
            subprocess.run([sys.executable, str(ROOT / "src/ablation_epc_evidence_control.py"),
                            "--variant", variant, "--train-fraction", str(fraction / 100),
                            "--data-path", str(ROOT / "data/new_MDdata.xlsx"),
                            "--scibert-dir", str(args.scibert.resolve()),
                            "--knowledge-dir", str(args.knowledge.resolve()),
                            "--skip-base", "--out-dir", str(run), "--fig-dir", str(run / "figures"),
                            "--seed", "42", "--prior-epochs", str(args.prior_epochs)], check=True)
if __name__ == "__main__":
    main()
