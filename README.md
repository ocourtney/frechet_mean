# Fréchet Mean Graph Computation

Code for computing barycenter, or Fréchet mean, graphs in the spectral domain, accompanying the paper **Fast Computation of the Barycenter Graph in the Spectral Domain**.

This repository contains Python implementations and Jupyter notebooks for reproducing experiments on stochastic block model data and primary-school temporal contact-network data.

## Overview

The core idea is to estimate a mean graph by minimizing an average spectral distance between a candidate weighted block graph and a sample of observed graphs. The implementation works with the leading eigenvalues of adjacency matrices and uses finite-difference gradient descent to optimize the block-density parameters.

The repository includes two variants:

- `frechetMeanGraph_constant_q.py`  
  Optimizes within-community block densities while using a fixed cross-community density `q`.

- `frechetMeanGraph_variable_q.py`  
  Optimizes both within-community block densities and the cross-community density `q`.

The notebooks use these helper functions to generate figures and experiments for the paper.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ocourtney/frechet_mean.git
cd frechet_mean
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The scripts were written as research code rather than as an installable Python package, so the simplest workflow is to run notebooks from the repository root.

---

## Notebooks

### `SBM_paperFigs.ipynb`

Experiments using synthetic stochastic block model graphs. Use this notebook to reproduce figures and examples based on generated SBM samples.

### `PS_paperFigs.ipynb`

Experiments using the primary-school temporal contact-network dataset from SocioPatterns.

This notebook expects the following local data files:

```text
primaryschool.csv
metadata_primaryschool.txt
```

These data files are **not included** in this repository. Download them from the SocioPatterns primary-school temporal network dataset and place them in the repository root before running the notebook.

---

## License

This project is released under the MIT License. See `LICENSE` for details.

---
