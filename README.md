# EINN · Original manuscript reproducibility archive

This repository contains research artifacts for *A Language Experience-Informed Neural Network for the Rheological Properties Prediction of Polymer Nanocomposites*. Its experiments predict log₁₀ viscosity from filler volume fraction, reduced temperature and log₁₀ shear rate. Table and figure numbers below refer to the matching author archive.

## Reproduce the archived results

Use Python 3.12. The Mac CPU dependency snapshot is in [environment/requirements-macos-cpu.lock](environment/requirements-macos-cpu.lock). Use four CPU threads and the pinned environment for checkpoint comparisons. A standard Python installation with compatible PyTorch, Transformers and scientific libraries can run the same scripts.

```bash
python -m pip install openpyxl numpy pandas matplotlib
python scripts/verify_archive.py
python scripts/render_figures.py reproduction-output
```

Run these commands from the repository root. The first script verifies the data checksums, recomputes Table 1 MAEs from held-out sample-level predictions, compares Table 4 with the archived summary, checks Figure 1 predictions and regenerates the four sets of training-derived prior strings. The second draws Figures 1–6 and 8 from their archived input tables. Figures 7 and 9 are conceptual artwork, supplied as PDFs.

## Replay the saved models

For these checks, also install PyTorch, Transformers, scikit-learn and tqdm using the tested versions in the environment lock. Torch loads checked model tensors on CPU. The scripts use local assets and do not call language model services.

```bash
python scripts/replay_evidence_head.py
python scripts/replay_checkpoint.py --weights-root PATH_TO_DEPOSIT/benchmark-3.11 --scibert PATH_TO_DEPOSIT/models/scibert_safe
```

The first command evaluates three KREPC correction heads using the archived numeric base predictions, scalers, bin parameters and frozen pooled encoder features. The second needs the version-matched weights deposit and recomputes the four Table 1 template-correction models on all stored held-out states, starting from the original numerical backbone and using their archived input templates. Neither command fits a new model. The small deposit can be used with the first command without fetching the full weights.

## Find a result

| Artifact | Location | Scope |
|---|---|---|
| 406 viscosity observations and supplement | `data/` | Exact source spreadsheet and numeric CSV. Source audit marks coordinate overlap with an earlier file. |
| Table 1 and training budget | `runs/benchmark/` | Four grouped splits, predictions, logs and prior banks. Training-derived templates are fixed per fold. |
| Table 4 and additional ablations | `runs/ablation/` | Summary and per-fraction metrics. A complete set of separate variant checkpoints has not been recovered. |
| 15% and extreme conditions | `runs/benchmark_sparsefractions_run/`, `runs/benchmark_extreme_15_25_50_75_100/` | Archived 15% split, logs and summary results. |
| KREPC retrieval and correction | `runs/krepc_cached_*/` | Numeric predictions, locators, source flags, scalers, frozen text-encoder features and correction head weights. |
| Original training and plotting software | `src/`, `figures/` | See the verified entry points above and [run notes](docs/run_notes.md) for the historical training interfaces. |
| Data lineage | `docs/artifact_provenance.json` | Input filename, transformation, and SHA-256 for each research artifact. |

The 406 observations are the processed inputs used for machine learning. The archived set does not contain a complete per-condition mapping to the original LAMMPS input decks, seed and initial configuration, trajectories, or stress time series. Figure 3a is reconstructed from the archived plotted coordinates. Code is provided with its original behavior and the verification scripts document what has been reproduced. Rights and redistribution are described in [run notes](docs/run_notes.md).
