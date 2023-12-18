#!/bin/sh -l

#SBATCH -J read                                                   # Job name
#SBATCH --mail-type=ALL                                           # Request status by email
#SBATCH -N 1                                                      # Total number of nodes requested
#SBATCH -n 1                                                      # Total number of cores requested
#SBATCH --get-user-env                                            # retrieve the users login environment
#SBATCH --mem=50G                                                 # server memory requested (per node)
#SBATCH -t 1:00:00                                                # Time limit (hh:mm:ss)
#SBATCH -o outputs/%x.out
#SBATCH -e errors/%x.err  

python ../read_experiments.py --exp_name "homogeneous-psi" --mode "inference" --experiment_dir "floods-Ida" --n_burnin "20000"