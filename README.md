# EINN for polymer nanocomposite rheology

EINN predicts steady-shear viscosity from filler volume fraction, reduced temperature and shear rate. The repository accompanies *A Language Experience-Informed Neural Network for the Rheological Properties Prediction of Polymer Nanocomposites*.

## Start here

Clone the repository and use Python 3.12 in a clean environment. These checks need no weights or API account.

```sh
python -m pip install openpyxl numpy pandas matplotlib
python scripts/verify_archive.py
python scripts/render_figures.py reproduction-output
```

The commands recompute Table 1 from the archived test predictions, check Table 4 against its recorded metrics, and draw Figures 1–6 and 8. Figures 7 and 9 are supplied as artwork. The [406 viscosity observations](data/new_MDdata_406_records.csv), [supporting workbook](data/Supplementary_Data.xlsx), and the [fixed partitions and predictions](runs/benchmark) are included.

## Weights and training

The files in [Releases](https://github.com/Bibiriosimon/einn-rheology/releases/tag/v1.0.0) contain the original inference tensors and SciBERT initializer. Each ZIP has an accompanying SHA-256. Start with the [run instructions](docs/reproduce.md) for checkpoint replay, retraining the March benchmark, or reproducing the evidence-based experiments.

| Location | Contents |
|---|---|
| [src](src) | TransRheo, template-based correction, retrieval-based correction and ablation code |
| [runs](runs) | Splits, predictions, metrics, scalers, prior templates and frozen evidence features |
| [figures](figures) | Source data, plotting scripts and reference PDFs |
| [nemd](nemd) | LAMMPS inputs and stress data for the six-condition reconstruction |
| [docs/verification](docs/verification) | Recorded checks and reproduction results |

The 406 training labels are processed viscosity values. See the [simulation record](nemd/README.md) for available parameters and the limits of tracing the historical labels to individual trajectories. See the [notice](NOTICE.md) for software, data and third-party assets.
