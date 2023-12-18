from datetime import datetime as dt
import pandas as pd
import networkx as nx
import geopandas as gpd
import time
start_time = time.time()

import sys
sys.path.append('../../d03_src/')
import model_MCMC as model
import process_graph as prg
import process_demographics as prd
import process_complaints as prc
import vars

import random

################################################################################
## SET THE HYPERPARAMETERS:

import argparse

parser = argparse.ArgumentParser()

#System outputs:
parser.add_argument("--j", type=int, default=random.randint(10*3, 10*4-1))
parser.add_argument("--results_folder", type=str, default=f'{vars._path_output_raw}inference/')
parser.add_argument("--test", default=False, action=argparse.BooleanOptionalAction)

#Experiments:
parser.add_argument("--dataset", type=str, default='floods')
parser.add_argument("--event", type=str, default='ida')
parser.add_argument("--duration", type=int, default=7)
parser.add_argument("--train_cutoff", type=str, default='duration')
parser.add_argument("--train_duration", type=float, default=4)
parser.add_argument("--train_percentage", type=float, default=0.08)

#Structural parameters:
parser.add_argument("--graph_type", type=str, default='census')
parser.add_argument("--grid_side_length", type=int, default=50)
parser.add_argument("--census_unit", type=str, default='tracts')
parser.add_argument("--geohash_precision", type=int, default=6)

#Configuring the graph:
parser.add_argument('--trim_graph_variables', nargs='+', default=['population', 'park_area', 'land_area'])
#Available trimming parameters are: ['population', 'park_area', 'land_area', 'edge_length', 'node_degree']
parser.add_argument("--max_degree", type=int, default=9)
parser.add_argument("--max_edgelength", type=float, default=2000) #meters
parser.add_argument("--min_pop", type=int, default=100)
parser.add_argument("--max_parkarea", type=float, default=0.75)
parser.add_argument("--min_landarea", type=float, default=1/3)

#Demographics:
parser.add_argument("--covariates", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--covariates_standardize", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--covariates_names', nargs='+', default=vars._covariates)
parser.add_argument("--projected_crs", type=str, default=vars._projected_crs)

#Priors:
parser.add_argument("--psi_Beta_mean", type=float, default=0.6)
parser.add_argument("--psi_Beta_strength", type=float, default=2)

parser.add_argument("--alpha_mean", nargs='+', default=[0.]+[0.]*len(vars._covariates))
parser.add_argument("--alpha_sd", nargs='+', default=[1.]+[0.5]*len(vars._covariates))

parser.add_argument("--theta0_mean", type=float, default=0.)
parser.add_argument("--theta0_sd", type=float, default=0.5)
parser.add_argument("--theta1_mean", type=float, default=0.1)
parser.add_argument("--theta1_sd", type=float, default=0.03)

#MCMC hyperparameters:
parser.add_argument("--MCMC_n_iter", type=int, default=100)
parser.add_argument("--MCMC_n_burnin", type=int, default=10)
parser.add_argument("--MCMC_thin_frac", type=float, default=0.5)

#SVE hyperparameters (always using SVE, with adaptive stepsize, and SW):
parser.add_argument("--SVE_n_iter", type=int, default=1)
parser.add_argument("--SVE_n_aux", type=int, default=50)
parser.add_argument("--SVE_initial_stepsize", type=float, default=0.05)

#Regression hyperparameters (always using regression if covariates are passed):
parser.add_argument("--Reg_warmup", type=int, default=50)

args = parser.parse_args()
print(args)

#Adjust the prior:
if len(args.alpha_mean) != len(args.covariates_names): args.alpha_mean = args.alpha_mean[:len(args.covariates_names)+1]
if len(args.alpha_sd) != len(args.covariates_names): args.alpha_sd = args.alpha_sd[:len(args.covariates_names)+1]

################################################################################
## GENERATE THE GRAPH:
assert args.graph_type.lower() in ['census', 'geohash', 'grid']

#The geohash graph:
if args.graph_type.lower() == 'geohash':
    print(f'\nUsing geohash with precision {args.geohash_precision}')
    ny_gdf, ny_graph = prg.generate_graph_geohash(precision=args.geohash_precision,
                                                  weights_type='rook',
                                                  remove_water=True if 'land_area' in args.trim_graph_variables else False,
                                                  remove_zeropop=True if 'population' in args.trim_graph_variables else False,
                                                  remove_parks=True if 'park_area' in args.trim_graph_variables else False,
                                                  tresh_water=args.min_landarea,
                                                  tresh_population=args.min_pop,
                                                  tresh_parkarea=args.max_parkarea,
                                                  enforce_connection=True)
#The census graph:
elif args.graph_type.lower() == 'census':
    print(f'\nUsing census {args.census_unit}')
    ny_gdf, ny_graph = prg.generate_graph_census(census_unit=args.census_unit,
                                                 state='NY',
                                                 counties=['New York','Bronx','Kings','Queens','Richmond'],
                                                 weights_type='rook',
                                                 remove_water=True if 'land_area' in args.trim_graph_variables else False,
                                                 remove_zeropop=True if 'population' in args.trim_graph_variables else False,
                                                 remove_parks=True if 'park_area' in args.trim_graph_variables else False,
                                                 tresh_population=args.min_pop,
                                                 tresh_parkarea=args.max_parkarea,
                                                 enforce_connection=True)
    
ny_gdf_proj = ny_gdf.to_crs(args.projected_crs)
print(f'Graph built with {ny_graph.order()} nodes')

################################################################################
#GET THE COVARIATES:

if args.covariates:
    covariates = prd.include_covariates(ny_gdf,
                                        col_to_merge_on='GEOID' if args.graph_type.lower()=='census' else 'geohash',
                                        processed_covariates_dir=vars._processed_covariates_dir,
                                        covariates_names=args.covariates_names,
                                        standardize=args.covariates_standardize)
    print(f"\nCovariates: {', '.join(args.covariates_names)}")
else:
    covariates = None

################################################################################
#GET THE COMPLAINTS:

if args.dataset == 'rats':
    path_311 = vars._path_311_rodents
    start_date = vars._rodents_start_date

elif args.dataset == 'floods':
    path_311 = vars._path_311_floods
    start_date = vars._floods_start_date[args.event.lower()]

train_T, test_T = prc.get_complaints(ny_gdf_proj.copy(),
                                     event_date=start_date,
                                     filter_mode=args.train_cutoff,
                                     event_duration=args.duration,
                                     train_duration=args.train_duration,
                                     train_percentage=args.train_percentage,
                                     return_test_complaints=True,
                                     path_311=path_311)

################################################################################
#Run the model:

HP = {'psi_method': 'Beta' if covariates is None else 'Reg',
      
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

sampled_A, _, sampled_params = model.run_MCMC(ny_graph,
                                              covariates=covariates,
                                              fixed_T=True,
                                              T=train_T,
                                              MCMC_iterations=args.MCMC_n_iter,
                                              MCMC_burnin=args.MCMC_n_burnin,
                                              MCMC_thinfrac=args.MCMC_thin_frac,
                                              MCMC_verbose=False,
                                              params_sample_dict=HP,
                                              debug=False)
#PU_assertion:
assert 0 not in sampled_A + train_T

################################################################################
#Merge output dataframes:
sampled_params_df = pd.DataFrame(sampled_params)
sampled_Adf = pd.DataFrame(sampled_A, columns=[f'A{k+1}' for k in range(len(sampled_A[0]))])
sampled_params_df = sampled_params_df.merge(sampled_Adf, left_index=True, right_index=True)

true_params_df = pd.DataFrame(train_T.reshape(1, len(train_T)), columns=[f'T{k+1}' for k in range(len(train_T))])
test_Tdf = pd.DataFrame(test_T.reshape(1, len(test_T)), columns=[f'test_T{k+1}' for k in range(len(test_T))])
true_params_df = true_params_df.merge(test_Tdf, left_index=True, right_index=True)

################################################################################
#SAVE OUTPUTS:

string_graph_name = f'{args.graph_type}'
string_covariate_name = 'homogeneous' if not args.covariates else ('multivariate' if len(args.covariates_names)==6
                                                                                  else '_'.join(args.covariates_names))
string_exp = f'{string_graph_name}-{string_covariate_name}'

if args.test:
    output_directory = f'{args.results_folder}test/'
    exp_name = f'{args.dataset}-{args.event}-{string_exp}'
else:
    output_directory = f'{args.results_folder}{args.dataset}-{args.event}/'
    exp_name = string_exp

try:        
    sampled_params_df.to_csv(f'{output_directory}sample-{exp_name}-{args.j}.csv')
    true_params_df.to_csv(f'{output_directory}true-{exp_name}-{args.j}.csv')
except: #in case the output directory doesnt exist
    sampled_params_df.to_csv(f'sample-{args.event}-{args.train_duration}-{args.j}.csv')
    true_params_df.to_csv(f'true-{args.event}-{args.train_duration}-{args.j}.csv')
    
################################################################################ 
run_time = time.time() - start_time
print(f'\nTotal runtime: {(run_time//3600):.0f} hours, {((run_time%3600)//60):.0f} minutes and {(run_time%60):.0f} seconds')