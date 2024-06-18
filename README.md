# Kinetics-informed Neural Networks modeling of TAP response
This repo contains the code for the paper *"Fitting micro-kinetic models to transient kinetics of temporal analysis of product reactors using kinetics-informed neural networks"* by D. Nai *et al*. This work aims to give a proof-of-concept demonstration of modeling TAP response through KINNs using carbon monoxide oxidation as the sample mechanism.

## Info
Authors: Dingqi Nai, Gabriel S. Gusmão, Zachary A. Kilwein, Fani Boukouvala, Andrew J. Medford
School of Chemical and Biomolecular Engineering, Georgia Institute of Technology
Email correspondence: [dnai3@gatech.edu](mailto:dnai3@gatech.edu), [ajm@gatech.edu](mailto:ajm@gatech.edu)

The following folders are included in this repo:
- *data*: The synthetic TAP response data, including the outlet flow *(TAP_experimental_data)*, catalyst zone concentration *(TAP_thin_data)*, catalyst zone flow *(TAP_cat_in/TAP_cat_out)*, and outlet flow moments *(TAP_moments)*.
- *tapsolver*: The TAPSolver script used to generate the synthetic TAP response.
- *pyomo*: The notebook uses pyomo.dae to model the TAP response.
- *kinns*: The notebooks use kinns to model the TAP response, including data preprocessing, modeling, and figure generation.

## Requirements
- [JAX](https://github.com/google/jax)
- [TAPSolver](https://github.com/medford-group/TAPsolver)
- [Pyomo](https://github.com/Pyomo/pyomo)
- [tapsap](https://github.com/IdahoLabResearch/tapsap)
