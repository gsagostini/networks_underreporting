#!/bin/sh -l

#SBATCH -J inference                                              # Job name
#SBATCH --mail-type=ALL                                           # Request status by email
#SBATCH -N 1                                                      # Total number of nodes requested
#SBATCH -n 1                                                      # Total number of cores requested
#SBATCH --get-user-env                                            # retrieve the users login environment
#SBATCH --mem=20G                                                 # server memory requested (per node)
#SBATCH -t 300:00:00                                              # Time limit (hh:mm:ss)
#SBATCH -o outputs/%x_%a.out
#SBATCH -e errors/%x_%a.err  
#SBATCH --array=1-3

python ../inference.py --j $SLURM_ARRAY_TASK_ID --dataset "floods" --event "ida"--MCMC_n_iter "60_000" --MCMC_n_burnin "20_000" --train_duration "1" --covariates
