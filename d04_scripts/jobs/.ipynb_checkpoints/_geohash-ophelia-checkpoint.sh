#!/bin/sh -l
    
#SBATCH -J inference-hash-ophelia                                 # Job name
#SBATCH --mail-type=ALL                                           # Request status by email
#SBATCH --mail-user=gs665@cornell.edu                             # Email address to send results to
#SBATCH -N 1                                                      # Total number of nodes requested
#SBATCH -n 1                                                      # Total number of cores requested
#SBATCH --get-user-env                                            # retrieve the users login environment
#SBATCH --mem=50G                                                 # server memory requested (per node)
#SBATCH -t 300:00:00                                              # Time limit (hh:mm:ss)
#SBATCH --partition=pierson                                       # Request partition
#SBATCH -w klara                                                  # Request node
#SBATCH -o outputs/%x_%a.out
#SBATCH -e errors/%x_%a.err  
#SBATCH --array=1-3

source /share/apps/anaconda3/2020.11/etc/profile.d/conda.sh
source activate /share/pierson/conda_virtualenvs/networks_underreporting_env/

python3 ../inference.py --j "$SLURM_ARRAY_TASK_ID" --graph_type 'geohash' --dataset "floods" --event 'ophelia' --train_cutoff "percentage" --train_percentage "0.08" --MCMC_n_iter "60_000" --MCMC_n_burnin "20_000" --covariates