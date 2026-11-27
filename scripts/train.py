"""Train the March benchmark using the readable sources."""
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import benchmark_llm_transrheo_datafractions as benchmark
from historical_template import save_final_outputs, train_llm_correction

def main():
    torch.set_num_threads(int(os.environ.get("EINN_CPU_THREADS", "4")))
    args = benchmark.parse_args()
    if args.out_root.exists():
        raise SystemExit(f"Choose a new output directory: {args.out_root}")
    benchmark.train_llm_correction = train_llm_correction
    benchmark.save_final_outputs = save_final_outputs
    benchmark.main()

if __name__ == "__main__":
    main()
