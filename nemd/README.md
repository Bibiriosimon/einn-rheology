# Molecular simulation inputs

EINN takes a [processed table of 406 reduced shear viscosities](../data/new_MDdata_406_records.csv) as its modeling input. The [coordinate comparison](../data/legacy_coordinate_audit.csv) identifies shared coordinates with an earlier file. See [Li et al., J. Phys. Chem. B 2023](https://doi.org/10.1021/acs.jpcb.3c01697) for the related physical study.

The input decks and stress series here belong to a separate six-condition reconstruction of the harmonic polymer–filler mixture from the [public preprint](https://doi.org/10.26434/chemrxiv-2023-1zx8l). The [configuration record](registration.json) gives the box volume, seed, thermostat and barostat settings, timestep and stress window. Paths in the input files are relative. Run preparation from each `preparation/nf*/` directory, then run equilibration and flow from their case directories. Restart files depend on the LAMMPS version. These runs used 30,000 polymer beads and took 9.56 million aggregate MD steps. LAMMPS needs harmonic bonds, `lj/expand` and compatible SLLOD options.

```bash
python nemd/recompute_viscosities.py
```

This computes the six viscosities from the included stress series. One of the four main comparison cases agreed with the corresponding archived viscosity within 10%. See [the four comparisons](comparison.csv) and [the two sensitivity controls](comparison.json).

The retained run used LAMMPS dated 30 September 2026 with KOKKOS. The flow decks specify SLLOD with `integrator legacy peculiar no kick yes` and zero-interval box deformation. To rerun, use a build that supports these settings and invoke `lmp -in in.run` in each stage directory. The archived GPU execution used `-k on g 1 -sf kk -pk kokkos gpu/aware off`. The preparation stage writes the restart consumed by both branches for that filler count. The reconstruction did not use supplier restarts.
