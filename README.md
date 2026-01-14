# interactive-profiles-ICL

[![License: GPL‑3.0](https://img.shields.io/badge/license-GPLv3-blue)](LICENSE)

Companion repository for the paper

**“How satellite populations and their orbits set the phase-space relationship between intracluster light and dark matter”**  
G. Martin et al., School of Physics & Astronomy, University of Nottingham

This repository contains the interactive figures and supporting analysis code used in the paper, including the modelling framework used to characterise how satellite properties map to the spatial and phase-space distribution of stripped stars and dark matter.

---

## Purpose

This repository supports reproducibility of the paper’s results by providing:

1. **Interactive figures**  
   Scripts that generate the plots used in the paper, visualising radial profiles and phase-space properties of stripped material.

2. **Stripped material modelling**  
   A multi-output Gaussian process model using a linear model of coregionalisation (LMC) to quantify relationships between satellite properties (mass ratio and orbital circularity) and:
   - Orbital energy and angular momentum of stripped stars and dark matter  
   - Stripped mass fractions  
   - Radial distributions of stripped material within the cluster  

   The modelling code, including inference configuration and validation procedures, is in the [`MCMC/`](https://github.com/garrethmartin/interactive-profiles-ICL/tree/main/MCMC) folder.

---

## Requirements

Dependencies:

- **Modelling code (`MCMC/`)**  
  The Bayesian modelling framework (PyMC and associated packages) has its own dependencies. A Conda environment specification is provided:

  ```bash
  conda env create -f MCMC/pymc_env.yml
  conda activate <env_name>
  ```

  This installs the packages needed to run the model code contained in [`MCMC/`](https://github.com/garrethmartin/interactive-profiles-ICL/tree/main/MCMC).

---

*This repository is a companion to the published paper, providing interactive visualisations and a Gaussian process framework for quantifying the spatial and phase-space properties of stripped stars and dark matter.*
