# Reproducing the original EINN experiments

## Model files

Use the assets linked from [the weight release](https://github.com/Bibiriosimon/einn-rheology/releases/tag/v1.0.0). Verify the accompanying SHA-256 files before unpacking. The examples below assume the following directories outside the Git checkout.

- `assets/table1` contains `benchmark-3.11/` and the original SciBERT vocabulary and configuration.
- `assets/scibert` contains the original pretrained `model.safetensors`, configuration and vocabulary used to initialize a new model.
- `assets/table4` contains the four ablation backbones.
- `assets/krepc` contains the three full KREPC weights. The much smaller correction heads used for frozen-feature replay are already in Git.

Use Python 3.12 and `python -m pip install -r environment/requirements.txt`. The [Mac CPU lock](../environment/requirements-macos-cpu.lock) records the versions used for the last local verification. For GPU training, use a PyTorch build compatible with the available CUDA driver. Refit results can vary with numerical libraries, devices and early stopping behavior. The original archived metrics are preserved separately from new runs.

## Table 1 and the 15% extension

Replay all four original backbone/correction pairs on their common set of 41 held-out states.

```sh
python scripts/replay_checkpoint.py --weights-root assets/table1/benchmark-3.11 \
  --scibert assets/table1/models/scibert_safe
```

To fit from the viscosity table, use the readable trainer. This runs the numerical backbone, residual prior bank, template-conditioned SciBERT correction and four classical regressors across the four budgets. The individual stage defaults are available from `--help`. The historical split uses seed 42. Keep the input row order.

```sh
python scripts/train.py --data-path data/new_MDdata.xlsx \
  --scibert-dir assets/scibert --out-root reproduction-output/table1
```

The same benchmark can use the recovered March bytecode by replacing `scripts/train.py` with `scripts/train_march.py`. The loader verifies the preserved module hashes before execution. The readable version reuses the matching original functions and restores the March-specific template training and result aggregation. A comparison at 25% over one training epoch found exact agreement on the prediction CSVs, training loss and 211 saved tensors. The two paths retain the archived snapshot behavior, which may differ across CPU and CUDA. The 15% study inputs and metrics are in `runs/benchmark_sparsefractions_run/15pct`. The 15% fraction was the third entry in a separate run. Use `--fractions 0.05,0.10,0.15` to regenerate that sequence. Pass the four main fractions in their original order for Table 1. `python scripts/verify_splits.py` checks all seven partitions against the archived record IDs.

## Retrieval and summary correction

The paper archive contains three variants with a common numerical starting point. The runs preserve locally constructed summaries, DeepSeek V4 Flash with 403 recorded provider responses and DeepSeek V4 Pro with 180. The remaining Flash and Pro cases retain their original deterministic fallback flags.

```sh
python scripts/replay_evidence_head.py
```

This command starts from the original pooled SciBERT features and base predictions. It replays the computation of the numerical correction and gate against all 406 rows for each variant. To rebuild retrieval and the original text before encoding, run the following with local copies of the original indexed passages and authorized commercial API caches.

```sh
python scripts/rebuild_evidence_inputs.py --knowledge path/to/Knowledge \
  --cache-root path/to/historical-runs
```

The script verifies the retrieved evidence order and checks SHA-256 fingerprints of the original query, assembled evidence and summary strings. It applies the archived fallback choice for each row and uses no API. With only the knowledge directory, it checks the deterministic variant. The index contains 4,137 formatted passages from Macosko and Ferry. The [run notes](run_notes.md) describe the licensed inputs and scope. The 1,218 audited tuples of text fingerprints and the source and page locators are in Git. New summaries from a live API may differ.

For a new fit with the local licensed knowledge directory, use the original KREPC training entry point. It uses the data supplied by `--data-path` and a numerical model output in `--base-predictions-csv` when using `--base-mode cached`. The archived retrieval uses four items, two per source, and evidence and summary truncation lengths of 320 and 192 tokens.

```sh
python src/llm_transrheo_prior_correction_pipeline.py \
  --data-path data/new_MDdata.xlsx --scibert-dir assets/scibert \
  --base-mode cached --base-predictions-csv runs/benchmark/100pct/transrheo_y_predictions_all_splits.csv \
  --knowledge-dir path/to/Knowledge --summary-backend deterministic \
  --prior-retrieval-topk 4 --prior-per-source-cap 2 --out-dir reproduction-output/krepc-det
```

The original online builder includes the request and formatting rules in source. It requires an authorized provider credential via `DEEPSEEK_API_KEY`; a fresh provider call is a new experiment. Keep any resulting verbatim material local unless you have redistribution rights.

## Table 4

The held-out split and cached TransRheo base predictions are included for each fraction. Each cell gets its own output directory so later variants cannot overwrite the weights or predictions of another run.

```sh
python scripts/refit_ablation.py --scibert assets/scibert \
  --knowledge path/to/Knowledge --base-checkpoints assets/table4 \
  --out reproduction-output/table4
```

The [ablation prediction audit](verification/ablation_prediction_audit.json) identifies which retained sample-level exports match an archived Table 4 variant by held-out MAE. Historical exports used shared names, so only the identifiable matches are treated as evidence for individual cells.

## Molecular simulation

The [NEMD record](../nemd/README.md) describes the published model parameters that could be recovered, the exact assumptions in the separate six-condition reconstruction and the available input decks. The archive cannot regenerate the original 406 supplied labels directly from raw trajectories.

## Testing with the example knowledge base

A six-note [example corpus](../examples/knowledge) exercises retrieval, text encoding and optimization without a publisher copy or API key. This runs the full 406-row input pipeline and one training epoch.

```sh
python scripts/refit_evidence.py --run krepc_cached_det_full --knowledge examples/knowledge \
  --scibert assets/scibert --epochs 1 --out reproduction-output/example
```

For a historical refit, change the knowledge directory to the original authorized local index and use the desired full training budget. The Flash and Pro variants additionally accept their respective authorized `--cache` files. Cache misses use the deterministic fallback, and this replay entry point cannot call the provider.

Use `python scripts/compare_refit.py reproduction-output/table1` to compare completed fresh fits with the original predictions on identical held-out rows. The [execution test record](verification/refit_execution_tests.json) reports the shorter end-to-end training checks separately. The [CPU refit record](verification/march_cpu_refit.json) lists the fractions run with the historical default training budget and their deviation from the original runs.

The full four-fraction CPU refit completed with the archived March training entry point. The classical baselines and both neural paths were fit again from the released labels and SciBERT initializer. The [CPU audit download](https://github.com/Bibiriosimon/einn-rheology/releases/download/v1.0.1/march-cpu-refit-audit.zip) contains the new held-out predictions, splits, logs and per-model comparisons. For the template model, the fresh log-viscosity MAEs at 25%, 50%, 75% and 100% were 0.21664, 0.14789, 0.13504 and 0.06400. The original checkpoint results were 0.20298, 0.21722, 0.13497 and 0.05763. The released original weights reproduce the latter results.
