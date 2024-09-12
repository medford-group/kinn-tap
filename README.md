# Kinetics-informed Neural Networks modeling of TAP response
This repo contains the code for the paper *"Micro-kinetic modeling of temporal analysis of products data using kinetics-informed neural networks"* by D. Nai *et al*. This work aims to give a proof-of-concept demonstration of modeling TAP response through KINNs using carbon monoxide oxidation as the sample mechanism.

## Info
Authors: Dingqi Nai, Gabriel S. Gusmão, Zachary A. Kilwein, Fani Boukouvala, Andrew J. Medford

School of Chemical and Biomolecular Engineering, Georgia Institute of Technology

Email correspondence: [dnai3@gatech.edu](mailto:dnai3@gatech.edu), [ajm@gatech.edu](mailto:ajm@gatech.edu)

The following folders are included in this repo:
- *data*: The synthetic TAP response data, including the outlet flow *(TAP_experimental_data)*, catalyst zone concentration *(TAP_thin_data)*, catalyst zone flow *(TAP_cat_in/TAP_cat_out)*, and outlet flow moments *(TAP_moments)*.
- *tapsolver*: The TAPSolver script used to generate the synthetic TAP response.
- *pyomo*: The notebook uses pyomo.dae to model the TAP response.
- *kinns*: The functional programmed kinns and the notebooks use kinns to model the TAP response, including data preprocessing, modeling, and figure generation.

## Dependencies

This project depends on the following packages:

- [JAX](https://github.com/google/jax) (>=0.4.21)
- [TAPSolver](https://github.com/medford-group/TAPsolver)
- [Pyomo](https://github.com/Pyomo/pyomo)
- [tapsap](https://github.com/IdahoLabResearch/tapsap)
- [SciPy](https://scipy.org/) (>=1.11.4)
- [Matplotlib](https://matplotlib.org/) (>=3.8.2)
- [pandas](https://pandas.pydata.org/) (>=2.1.4)

## Installation

We strongly recommend installing TAPSolver, Pyomo, and JAX in separate environments to avoid potential compatibility issues. Installation guides for [JAX](https://github.com/google/jax), [Pyomo](https://github.com/Pyomo/pyomo), [SciPy](https://scipy.org/),
[Matplotlib](https://matplotlib.org/), and [pandas](https://pandas.pydata.org/) can be found on their websites. This guide will only cover the installation of [TAPSolver](https://github.com/medford-group/TAPsolver) and [tapsap](https://github.com/IdahoLabResearch/tapsap).

### TAPSolver
  
Please note that FEniCS is not pre-built for Windows. Windows users please use the Windows Subsystem for Linux (WSL) to run the code.

We recommend using the 'thinzoneFlux' branch of TAPSolver to directly obtain the thin zone flux. For other branches, the thin zone flux can be calculated using the returned mesh and concentration profiles.
  
```bash
conda create -n tapsolver -c conda-forge/label/cf202003 fenics
conda activate tapsolver
pip install --upgrade git+https://github.com/medford-group/TAPsolver.git@thinzoneFlux
pip install --upgrade git+https://github.com/dolfin-adjoint/pyadjoint.git@faster-ufl
pip install --upgrade git+git://github.com/DDPSE/PyDDSBB/
pip install CheKiPEUQ[COMPLETE]
pip install geneticalgorithm
```

### tapsap

```bash
pip install --upgrade pip
pip install --upgrade git+https://github.com/IdahoLabResearch/tapsap.git
```
