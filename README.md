# A Bayesian Spatial Model to Correct Under-Reporting in Urban Crowdsourcing

Code to accompany AAAI 24 paper. Written by Gabriel S. Agostini, Nikhil Garg, and Emma Pierson.

## Installation

A `.yml` file is provided with all required packages and their versions.

## Repository Setup

```
│
├── d01_data
│   ├── d01_raw                                   <- Contain raw data files that are publicly available        
│   └── d02_processed                             <- Contain files produced by this code (census demographic covariates)
│
├── d02_notebooks                                 <- Jupyter notebooks that produce plots used in the submission
│   ├── 1_preprocessing_311_complaints.ipynb      <- Basic EDA of 311 flood reports
│   ├── 2_preprocessing_demographic_data.ipynb    <- Shows processing of Census and ACS data into demographic features
│   ├── 3_model_example.ipynb                     <- Read inference on real world summaries and produce visualizations
│   └── 4_producing_results.ipynb                 <- Read semi-synthetic simulation summaries and produce visualizations
│
├── d03_src                                       <- Source code for use in this project, which can be imported as modules into the notebooks and scripts
│   ├── model_*.py                                <- Model functions     
│   ├── evaluate_*.py                             <- Misc. functions for reading model results and visualization
│   ├── process_*.py                              <- Misc. functions for data pre-processing
│   └── vars.py                                   <- Main variables used in other scripts such as paths and flood event dates
│
├── d04_scripts                                   <- Full code routines
│   ├── calibrate.py                              <- Semi-synthetic data experiments to test model calibration
│   ├── generate.py                               <- Generates data to test priors
│   ├── inference.py                              <- Empirical routines
│   └── read_inferences.py                        <- Read job outputs into posterior and summaries            
│
├── d05_joboutputs                                <- Outputs should be directed and read from this directory. Not included due to size limits
│
├── d06_processed-outputs                         <- Outputs from the read_inference.py script such as posterior summaries and inferred quantities
│
└── d07_plots                                     <- Plots used on paper
```

## Usage

The main function we implement is `d03_src.model_MCMC.run_MCMC` which runs the chain to either sample the variables given fixed parameters (pass the argument `fixed_params=True` along with a dictionary `params`) or sample the parameters given observed reports $T$ (pass the argument `fixed_T=True` along with an array `T`). A `NetworkX.Graph` must be passed. Other arguments that control the chain hyperparameters (number of iterations, thinning fraction, burn-in period) or the sampling hyperparameters (step-size on the SVEA, priors) are detailed in the signature.

The notebook `d02_notebooks/3_model_example.ipynb` details the usage of these functions.