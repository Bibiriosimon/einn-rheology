# Experimental records

## Model roles

- Table 1 pairs the numerical TransRheo backbone with a template-based residual correction. The stored prior banks reproduce the templates and record the training and validation residual procedure. See the [run instructions](reproduce.md) for both March training entry points and the [verification records](verification) for their comparison.
- Table 4 uses deterministic summaries. The ablation runner and recorded metrics are included. The original shell runner shares output directories. Use one directory per variant when fitting again.
- The deterministic, DeepSeek Flash and DeepSeek Pro KREPC runs each have their own checkpoint and evidence locators, with record IDs, ranks and pages. Offline replay starts from the stored encoder features. To reproduce the earlier retrieval and summaries, use the original source index and, for the two API variants, the original response caches and provider settings.

## Data

The modeling table is provided as `data/new_MDdata.xlsx` and `data/new_MDdata_406_records.csv`. Its columns are filler volume fraction, reduced temperature, log shear rate and log viscosity. It contains 406 distinct input triples across 105 (filler fraction, temperature) groups. The [coordinate comparison](../data/legacy_coordinate_audit.csv) identifies 186 triples shared with an earlier 195-row file. Source attribution, simulation settings and redistribution rights for the other 220 triples require documentation from the data provider. See [Li et al., J. Phys. Chem. B 2023](https://doi.org/10.1021/acs.jpcb.3c01697) for the related physical study.

The weight release includes the configuration and vocabulary from [AllenAI SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased). See the [license notice](../NOTICE.md) for third-party terms. The evidence records give bibliography and passage locators together with the features used for offline replay. Use independently acquired, authorized copies of the underlying passages for retrieval and re-encoding.
