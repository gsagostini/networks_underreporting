#!/bin/sh -l

#SBATCH -J generate_tracts                                # Job name
#SBATCH --mail-type=ALL                                   # Request status by email
#SBATCH -N 1                                              # Total number of nodes requested
#SBATCH -n 21                                             # Total number of cores requested
#SBATCH --get-user-env                                    # retrieve the users login environment
#SBATCH --mem=100G                                        # server memory requested (per node)
#SBATCH -t 600:00:00                                      # Time limit (hh:mm:ss)
#SBATCH -o outputs/%x.out
#SBATCH -e errors/%x.err

python ../generate.py --graph_type "nyc_edge_trim" --nyc_unit "tracts"