"""Fit KREPC from the stored original baseline and local, authorized evidence inputs."""
import argparse
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import llm_transrheo_prior_correction_pipeline as pipeline

PROVIDERS = {
    "krepc_cached_det_full": None,
    "krepc_cached_ds_flash_full": "deepseek-v4-flash",
    "krepc_cached_ds_full": "deepseek-v4-pro",
}

class ArchivedCacheMiss(Exception):
    """The historical run used the local deterministic summary for this request."""

def use_fallback(**kwargs):
    raise ArchivedCacheMiss("Offline replay")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, choices=PROVIDERS)
    parser.add_argument("--knowledge", required=True, type=Path)
    parser.add_argument("--scibert", required=True, type=Path)
    parser.add_argument("--cache", type=Path, help="Original cache file for a provider variant")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    provider = PROVIDERS[args.run]
    if provider and (not args.cache or not args.cache.is_file()):
        parser.error("Provider replay requires the original authorized cache")
    if not (args.knowledge / "chunks_upgraded.jsonl").is_file():
        parser.error("No chunks_upgraded.jsonl under --knowledge")

    args.out.mkdir(parents=True, exist_ok=False)
    options = [
        "refit_evidence",
        "--data-path", str(ROOT / "data/new_MDdata.xlsx"),
        "--base-mode", "cached", "--base-predictions-csv",
        str(ROOT / "runs/benchmark/100pct/transrheo_y_predictions_all_splits.csv"),
        "--scibert-dir", str(args.scibert.resolve()),
        "--knowledge-dir", str(args.knowledge.resolve()),
        "--out-dir", str(args.out.resolve()), "--prior-epochs", str(args.epochs),
        "--prior-retrieval-topk", "4", "--prior-per-source-cap", "2",
    ]
    if provider:
        shutil.copyfile(args.cache, args.out / "krepc_summary_cache_deepseek.jsonl")
        # Preserve the original cache interface, while proving that a cache miss cannot make an API call.
        options += ["--summary-backend", "deepseek", "--deepseek-model", provider,
                    "--deepseek-api-key", "offline-no-provider-call", "--summary-allow-fallback"]
        pipeline.call_deepseek_summary_with_retry = use_fallback
    else:
        options += ["--summary-backend", "deterministic"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    pipeline.torch.set_num_threads(int(os.environ.get("EINN_CPU_THREADS", "4")))
    sys.argv = options
    pipeline.main()

if __name__ == "__main__":
    main()
