import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import random
import networkx as nx
from scipy.stats import norm
import os
import time
start_time = time.time()

from pygris import tracts, blocks
import geopandas as gpd
from libpysal import weights

import sys
sys.path.append('../../d03_src/')
import model_MCMC
from process_demographics import include_covariates
from process_graph import generate_graph_census, generate_graph_grid
from vars import _covariates, _projected_crs, _path_output_raw

################################################################################
## SET THE HYPERPARAMETERS:

import argparse

parser = argparse.ArgumentParser()

#System outputs:
parser.add_argument("--exp_idx", type=int, default=random.randint(10*3, 10*4-1))
parser.add_argument("--exp_name", type=str, default='') #subfolder
parser.add_argument("--results_folder", type=str, default=f'{_path_output_raw}calibration/')

#Experiments:
parser.add_argument("--number_chains", type=int, default=2)
parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--return_extra", type=str, default="")    # empty string for nothing, debug, or stepsize

#If we must use ground truth from a fixed experiment:
parser.add_argument("--repeat_trueparams", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--repeat_trueparams_dir", type=str, default="regression-psi-all6")

#Structural parameters:
parser.add_argument("--graph_type", type=str, default='nyc_edge_trim')  # grid, nyc, nyc_trim, nyc_edge_trim, nyc_clean 
parser.add_argument("--grid_l", type=int, default=100)
parser.add_argument("--nyc_unit", type=str, default='tracts')
parser.add_argument("--nyc_trim_max_degree", type=int, default=9)
parser.add_argument("--nyc_trim_max_edgelength", type=float, default=2000) #meters
parser.add_argument("--nyc_clean_min_pop", type=int, default=1)
parser.add_argument("--nyc_clean_max_parkarea", type=float, default=0.75)

#Demographics:
parser.add_argument("--covariates", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--covariates_standardize", default=True, action=argparse.BooleanOptionalAction) #Whether to standardize covariates
parser.add_argument('--covariates_names', nargs='+', default=_covariates)
parser.add_argument("--projected_crs", type=str, default=_projected_crs)

#Priors:
parser.add_argument("--psi_Beta_mean", type=float, default=0.6)
parser.add_argument("--psi_Beta_strength", type=float, default=2)

parser.add_argument("--alpha_mean", type=list, default=[0.]+[0.]*len(_covariates))
parser.add_argument("--alpha_sd", type=list, default=[1.]+[0.5]*len(_covariates))

parser.add_argument("--theta0_mean", type=float, default=0.)
parser.add_argument("--theta0_sd", type=float, default=0.5)
parser.add_argument("--theta1_mean", type=float, default=0.1)
parser.add_argument("--theta1_sd", type=float, default=0.03)

#MCMC hyperparameters:
parser.add_argument("--sample_n_burnin", type=int, default=1_000)
parser.add_argument("--MCMC_n_iter", type=int, default=30_000)
parser.add_argument("--MCMC_n_burnin", type=int, default=10_000)
parser.add_argument("--MCMC_thin_frac", type=float, default=0.5)

#SVE hyperparameters (always using SVE, with adaptive stepsize, and SW):
parser.add_argument("--SVE_n_iter", type=int, default=1)
parser.add_argument("--SVE_n_aux", type=int, default=50)
parser.add_argument("--SVE_initial_stepsize", type=float, default=0.05)

#Regression hyperparameters (always using regression if covariates are passed):
parser.add_argument("--Reg_warmup", type=int, default=50)

args = parser.parse_args()

print(args)

################################################################################
## GENERATE THE GRAPH:

if args.graph_type[:4] != 'nyc_': args.graph_type = 'nyc_' + args.graph_type
assert args.graph_type in ['nyc_full', 'nyc_trim', 'nyc_clean', 'nyc_edge_trim']

args.nyc_unit = args.nyc_unit.replace(" ", "")
assert args.nyc_unit in ['tracts', 'blocks', 'blockgroups']
    
ny_gdf, ny_graph, ny_gdf_raw = generate_graph_census(census_unit=args.nyc_unit,
                                                     state='NY',
                                                     counties=['New York','Bronx','Kings','Queens','Richmond'],
                                                     weight_scheme='rook',
                                                     remove_high_degree_nodes=True if args.graph_type == 'nyc_trim' else False,
                                                     remove_long_edges=True if args.graph_type == 'nyc_edge_trim' else False,
                                                     remove_parks=True if args.graph_type == 'nyc_clean' else False,
                                                     remove_zeropop=True,
                                                     tresh_degree=args.nyc_trim_max_degree,
                                                     tresh_edgelength=args.nyc_trim_max_edgelength,
                                                     tresh_population=args.nyc_clean_min_pop,
                                                     tresh_parkarea=args.nyc_clean_max_parkarea)
ny_gdf_proj = ny_gdf.to_crs(args.projected_crs)
ny_graph = nx.convert_node_labels_to_integers(ny_graph)

print(f'\n Graph built with {ny_graph.order()} nodes')

#GET THE COVARIATES:

if args.covariates:
    covariates = include_covariates(ny_gdf_proj,
                                    covariates_names=args.covariates_names,
                                    standardize=args.covariates_standardize)
    print(f"\n Covariates: {', '.join(args.covariates_names)}")
else:
    covariates = None
################################################################################
## SAMPLE THE DATA:

if args.repeat_trueparams:
    #Load the file:
    true_params_file = f'{args.results_folder}{args.repeat_trueparams_dir}/true-{args.exp_idx}.csv'
    true_params_df_full = pd.read_csv(true_params_file).drop('Unnamed: 0',axis=1)
    #Get the T values (from the first chain):
    T_cols = [col for col in true_params_df_full.columns if 'T' in col]
    T = true_params_df_full.loc[0, T_cols].values
else:
    #Generate theta and alpha:
    theta0 = norm.rvs(loc=args.theta0_mean, scale=args.theta0_sd, size=1) 
    theta1 = norm.rvs(loc=args.theta1_mean, scale=args.theta1_sd, size=1)
    theta_dict = {'theta0': theta0, 'theta1': theta1}
    alpha_dict = {f'alpha{k}': norm.rvs(loc=args.alpha_mean[k], scale=args.alpha_sd[k], size=1) for k in range(args.covariates*len(args.covariates_names)+1)}
    
    true_parameters = theta_dict|alpha_dict
    true_params_df = pd.DataFrame(true_parameters)
    
    #Generate A and T:
    A_list, T_list, _ = model_MCMC.run_MCMC(ny_graph,
                                            covariates=covariates,
                                            fixed_params=True,
                                            MCMC_iterations=args.sample_n_burnin,
                                            params=true_parameters)
    A = A_list[-1]
    T = T_list[-1]
    
    #Log the sampled ground truth:
    true_Adf = pd.DataFrame(A.reshape(1, len(A)), columns=[f'A{k+1}' for k in range(len(A))])
    true_Tdf = pd.DataFrame(T.reshape(1, len(T)), columns=[f'T{k+1}' for k in range(len(T))])
    true_params_df = true_params_df.merge(true_Adf, left_index=True, right_index=True).merge(true_Tdf, left_index=True, right_index=True)
    
    #Index the tru parameter dataframe as the chain:
    true_params_df_full = pd.concat([true_params_df]*args.number_chains).reset_index(drop=True)
    true_params_df_full.insert(0, 'chain', range(1, args.number_chains+1))

print('\n Data sampled')
################################################################################
## SAMPLE THE PARAMETERS:

HP = {'psi_method': 'Beta' if not args.covariates else 'Reg',
      
      'psi_Beta_priormean': args.psi_Beta_mean,
      'psi_Beta_priorstrength': args.psi_Beta_strength,

      'psi_Reg_warmup': args.Reg_warmup,
      'psi_Reg_priormean': args.alpha_mean,
      'psi_Reg_priorsd': args.alpha_sd,
      
      'theta_method':'SVE',
      'theta_SVE_warmup': args.SVE_n_iter,
      'theta_SVE_auxiliary': args.SVE_n_aux,
      'theta_SVE_proposalsigma': args.SVE_initial_stepsize,
      
      'theta_SVE_adaptive': True,
      'theta_SVE_adaptiveiter': 50,
      'theta_SVE_adaptivebounds':(0.25, 0.6),
      
      'theta_SVE_priormean':[args.theta0_mean, args.theta1_mean],
      'theta_SVE_priorsigma':[args.theta0_sd, args.theta1_sd]}

sampled_params_df_list = []
for chain in range(args.number_chains):
    
    chain_start_time = time.time()
    
    np.random.seed()
    
    sampled_A, sampled_T, sampled_params = model_MCMC.run_MCMC(ny_graph,
                                                               covariates=covariates,
                                                               fixed_T=True,
                                                               T=T,
                                                               MCMC_iterations=args.MCMC_n_iter,
                                                               MCMC_burnin=args.MCMC_n_burnin,
                                                               MCMC_thinfrac=args.MCMC_thin_frac,
                                                               MCMC_verbose=False,
                                                               params_sample_dict=HP,
                                                               debug=False)
        
    sampled_params_df = pd.DataFrame(sampled_params)
    sampled_Adf = pd.DataFrame(sampled_A, columns=[f'A{k+1}' for k in range(len(sampled_A[0]))])
    sampled_params_df = sampled_params_df.merge(sampled_Adf, left_index=True, right_index=True)
    sampled_params_df.insert(0, 'chain', chain+1)
    sampled_params_df_list.append(sampled_params_df)
    
    chain_run_time = time.time() - chain_start_time 
    print(f'\n Chain number {chain+1}/{args.number_chains} sampled in: {(chain_run_time//3600):.0f} hours, {((chain_run_time%3600)//60):.0f} minutes and {(chain_run_time%60):.0f} seconds')

sampled_params_df_full = pd.concat(sampled_params_df_list)

################################################################################
## SAVE:

try:
    output_dir = f'{args.results_folder}{args.exp_name}/'
    true_params_df_full.to_csv(f'{output_dir}true-{args.exp_idx}.csv')
    sampled_params_df_full.to_csv(f'{output_dir}sample-{args.exp_idx}.csv')
    print(f'\nSaved to {output_dir}')
    
except:
    true_params_df_full.to_csv(f'true-{args.exp_idx}.csv')
    sampled_params_df_full.to_csv(f'sample-{args.exp_idx}.csv')
    
################################################################################ 
run_time = time.time() - start_time
print(f'\nTotal runtime: {(run_time//3600):.0f} hours, {((run_time%3600)//60):.0f} minutes and {(run_time%60):.0f} seconds')