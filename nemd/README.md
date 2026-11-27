# Molecular simulation inputs

The training target in EINN is a supplied table of reduced shear viscosities. The related study is [Li et al., J. Phys. Chem. B 2023](https://doi.org/10.1021/acs.jpcb.3c01697). We retain the 406 values as used in the original machine learning work. The [coordinate audit](../data/legacy_coordinate_audit.csv) records which values overlap the earlier file. Individual trajectories and parameter decks for the supplied table were unavailable.

This directory documents a separate, six-condition reconstruction of the harmonic polymer–filler mixture from the [public preprint](https://doi.org/10.26434/chemrxiv-2023-1zx8l). The registration records the assumed box volume, seed, thermostat/barostat, timestep, and stress window. Paths in the reconstructed input files are relative. Run preparation from each `preparation/nf*/` directory, then the equilibration and flow decks from their respective case directories. Generated restart files are version-dependent. The original development runs used 30,000 polymer beads and took 9.56 million aggregate MD steps. The numerical engine requires harmonic bonds, `lj/expand`, and compatible SLLOD options.

```bash
python nemd/recompute_viscosities.py
```

This reproduces the six viscosities directly from the supplied stress series. The four primary comparison states are listed in `comparison.csv`. Only one of the four agreed with the corresponding archived viscosity within 10%. The two sensitivity controls are in `comparison.json`. These runs are separate from the labels used to train EINN.
