# EINN for polymer nanocomposite rheology

EINN predicts steady-shear viscosity from filler volume fraction, reduced temperature and shear rate. The repository accompanies *A Language Experience-Informed Neural Network for the Rheological Properties Prediction of Polymer Nanocomposites*.

## Start here

Clone the repository and use Python 3.12 in a clean environment. These checks need no weights or API account.

```sh
python -m pip install openpyxl numpy pandas matplotlib
python scripts/verify_archive.py
python scripts/render_figures.py reproduction-output
```

The first script recomputes Table 1 from the test predictions and checks Table 4 against the recorded metrics. The second draws Figures 1–6 and 8 from their source data. Figures 7 and 9 are supplied as artwork.

The modeling inputs are [406 processed reduced-viscosity values](data/new_MDdata_406_records.csv). See the [supporting workbook](data/Supplementary_Data.xlsx) and [benchmark partitions and predictions](runs/benchmark) for the data used in the paper.

## Weights and training

The [weight release](https://github.com/Bibiriosimon/einn-rheology/releases/tag/v1.0.0) contains the trained checkpoints and the SciBERT initializer. Each ZIP has a SHA-256 checksum. The [run instructions](docs/reproduce.md) cover replaying the reported predictions, fitting the models and running the text correction.

| Location | Contents |
|---|---|
| [src](src) | TransRheo, template-based correction, retrieval-based correction and ablation code |
| [runs](runs) | Splits, predictions, metrics, scalers, prior templates and frozen evidence features |
| [figures](figures) | Source data, plotting scripts and reference PDFs |
| [nemd](nemd) | LAMMPS inputs and stress data for the six-condition reconstruction |
| [docs/verification](docs/verification) | Recorded checks and reproduction results |

The [simulation notes](nemd/README.md) describe the six LAMMPS cases and their relationship to the modeling table. Software and third-party assets are described in the [license notice](NOTICE.md).
