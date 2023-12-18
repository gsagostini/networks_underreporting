#!/bin/bash -l

#SBATCH -J calibrate-homogeneous                   # Job name
#SBATCH --mail-type=ALL                            # Request status by email
#SBATCH -N 1                                       # Total number of nodes requested
#SBATCH -n 1                                       # Total number of cores requested
#SBATCH --get-user-env                             # Retrieve the users login environment
#SBATCH --mem=10G                                  # Server memory requested (per node)
#SBATCH -t 600:00:00                               # Time limit (hh:mm:ss)
#SBATCH -o outputs/%x_%a.out
#SBATCH -e errors/%x_%a.err
#SBATCH --array=1-100

python ../calibrate.py --j "$SLURM_ARRAY_TASK_ID" --exp_name "" --MCMC_n_iter "60_000" --MCMC_n_burnin "20_000" --covariates