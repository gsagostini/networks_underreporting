#######################################################################################################################
#This script reads the empirical results
#######################################################################################################################
import pandas as pd
import numpy as np
import arviz as az
from scipy.special import expit
import os

import sys
sys.path.append('../d03_src/')
import vars
import process_graph
import process_demographics
import model_baselines as base
import model_pooling as pool
import evaluate_utils as ev

#######################################################################################################################
## SET THE HYPERPARAMETERS:

import argparse
parser = argparse.ArgumentParser()

#Output directories:
parser.add_argument("--outputs_raw_dir",       type=str, default=f'{vars._path_output_raw}')
parser.add_argument("--outputs_processed_dir", type=str, default=f'{vars._path_output_processed}')
parser.add_argument("--mode",                  type=str, default='inference')
                    
#Storms, units, and models:
parser.add_argument('--storms', nargs='+', default=['henri', 'ida', 'ophelia'])
parser.add_argument('--units', nargs='+', default=['census', 'geohash'])
parser.add_argument("--univariates", default=False, action=argparse.BooleanOptionalAction)

#Covariates:
parser.add_argument('--covariates_to_include', nargs='+', default=vars._covariates_for_analysis)

#Configuring the graph:
parser.add_argument('--trim_graph_variables', nargs='+', default=['population', 'park_area', 'land_area']) #Available trimming parameters are: ['population', 'park_area', 'land_area']
parser.add_argument("--min_pop",      type=int, default=100)
parser.add_argument("--max_parkarea", type=float, default=3/4)
parser.add_argument("--min_landarea", type=float, default=1/3)

#MCMC hyperparameters:
parser.add_argument("--MCMC_n_burnin", type=int, default=10_000)
parser.add_argument("--MCMC_n_burnin_multivariate", nargs='?', const=-1, type=int)   #The number of burnin iterations on multivariate chains if different

#Priors
parser.add_argument("--psi_prior_mean", type=float, default=0.)
parser.add_argument("--psi_prior_sd", type=float, default=0.5)

args = parser.parse_args()
if args.MCMC_n_burnin_multivariate is None: args.MCMC_n_burnin_multivariate = args.MCMC_n_burnin
print(args)

#######################################################################################################################

for unit in args.units:

    print('UNIT:', unit)
    
    #Collect the graph:
    if unit == 'census':
        gdf, graph = process_graph.generate_graph_census(remove_zeropop=True if 'population' in args.trim_graph_variables else False,
                                                         tresh_population=args.min_pop,
                                                         remove_parks=True if 'park_area' in args.trim_graph_variables else False,
                                                         tresh_parkarea=args.max_parkarea,
                                                         remove_water=True if 'land_area' in args.trim_graph_variables else False)
    elif unit == 'geohash':
        gdf, graph = process_graph.generate_graph_geohash(precision=6,
                                                          remove_zeropop=True if 'population' in args.trim_graph_variables else False,
                                                          tresh_population=args.min_pop,
                                                          remove_parks=True if 'park_area' in args.trim_graph_variables else False,
                                                          tresh_parkarea=args.max_parkarea,
                                                          remove_water=True if 'land_area' in args.trim_graph_variables else False,
                                                          tresh_water=args.min_landarea)

    #Collect the covariates we used in the multivariate model:
    X_multivariate = process_demographics.include_covariates(gdf,
                                                             col_to_merge_on='GEOID' if unit=='census' else 'geohash',
                                                             processed_covariates_dir=vars._processed_covariates_dir,
                                                             covariates_names=vars._covariates,
                                                             standardize=True)

    #Collect covariates we would like to use in the equity analysis:
    X_to_include = process_demographics.include_covariates(gdf,
                                                           col_to_merge_on='GEOID' if unit=='census' else 'geohash',
                                                           processed_covariates_dir=vars._processed_covariates_dir,
                                                           covariates_names=args.covariates_to_include,
                                                           standardize=False)
    #We convert those to raw counts (and update the names):
    for idx, col in enumerate(args.covariates_to_include):
        if 'pct' in col: X_to_include[:,idx] = (X_to_include[:,0]*X_to_include[:,idx]).astype(int)
    X_to_include_names = [vars._covariates_for_analysis_renaming[c] if c in vars._covariates_for_analysis_renaming else c for c in args.covariates_to_include]
        
    #For the pooled model, we will collect summaries per parameter:
    posteriors_multivariate = {f'alpha{k+1}':[] for k,_ in enumerate(vars._covariates)}
    summaries_multivariate = []

    #Iterate through the three storms:
    for event in args.storms:
        print(' EVENT:', event)
        outputs_dir = f'{args.outputs_raw_dir}{args.mode}/floods-{event}/'
        files = os.listdir(outputs_dir)

        #First we look at the two full models:
        node_df = None
        for model in ['multivariate', 'homogeneous']:
            print('  MODEL:', model)
            
            #Let's collect all csv files corresponding to the model:
            file_paths = [f'{outputs_dir}{file}' for file in files if f'-{unit}-{model}-' in file]
            
            #The true dataframes contain training and test reports. For a sanity check, we verify they are
            #  all equal and collect just the first one:
            true_dfs = [pd.read_csv(file).drop('Unnamed: 0', axis=1) for file in file_paths if 'true' in file]
            assert all(true_dfs[0].equals(df) for df in true_dfs)
            true_df_model = pd.DataFrame(true_dfs[0].values.reshape((2,-1)), index=['T_train', 'T_test']).T
        
            #In case we have not yet defined the true df, we build the true dataframe and include our
            #  baselines:
            if node_df is None:
                #Initialize the model:
                node_df = true_df_model
                #Add the demographics we will analyze:
                node_df[X_to_include_names] = X_to_include
                #With the reports data, we can include the spatial baseline and the GP baseline:
                node_df['A_prob_trivial'] = base.get_trivial_probability(node_df['T_train'].values, graph)
                node_df['A_prob_GP'] = base.get_GP_probability(node_df['T_train'].values, gdf)
                #The report probability is akin to the flood probability:
                node_df['T_prob_GP'] = node_df['A_prob_GP']
                node_df['T_prob_trivial'] = node_df['A_prob_trivial']
                
            #In case we already have the true df, we can verify that we have the correct test/train sets:
            else: assert true_df_model.equals(node_df[['T_train', 'T_test']])
                
            #The sample dfs correspond to all chains we run in the model.
            sample_dfs = [pd.read_csv(file).drop('Unnamed: 0', axis=1) for file in file_paths if 'sample' in file]
            _ = [df.insert(0, 'chain', k+1) for k, df in enumerate(sample_dfs)]
            
            #After we crop the posterior distributions, we can concatenate all these chains:
            burnin = args.MCMC_n_burnin_multivariate if model=='multivariate' else args.MCMC_n_burnin
            posterior_dfs = [df.iloc[burnin:] for df in sample_dfs]
            posterior_df = pd.concat(posterior_dfs, ignore_index=True)
            
            #The posterior dataframe should have information on ground-truth states (A) and latent
            #  parameter (alphas and thetas). Let's get the column names corresponding to each of
            #  these groups:
            cols_A = [col for col in posterior_df.columns if 'A' in col]
            cols_params = [col for col in posterior_df.columns if col not in cols_A and col != 'chain']
            if 'alpha0' in cols_params and 'psi' in cols_params: cols_params.remove('psi')
        
            #Collect and save the posterior:
            posterior = posterior_df[cols_params]
            posterior.to_csv(f'{args.outputs_processed_dir}{args.mode}/model-posteriors/{unit}-{event}-{model}.csv', index=False)

            #Compute the r_hats
            r_hats = ev.compute_rhats(posterior_df, parameter_cols=cols_params, chain_col='chain')
            
            #We also save the posterior summaries which are easier to load:
            summary_df = posterior.describe(percentiles=[0.025, 0.25, 0.5, 0.75, 0.975]).T
            summary_df['r_hat'] = r_hats
            summary_df.index.name = 'Parameter'
            summary_df.to_csv(f'{args.outputs_processed_dir}{args.mode}/model-posteriors/summaries/{unit}-{event}-{model}.csv')
        
            #The A columns should be summarized (mean and Bernoulli 95% est. CI) and added to the
            #  node features dataframe:
            p = posterior_df[cols_A].mean().values/2 + 0.5
            node_df[f'A_prob_{model}'] = p
            node_df[f'A_prob_CIwidth_{model}'] = 1.96*np.sqrt(p*(1-p)/len(p))

            #Finally, we can include the estimated report rate per node:
            #In the homogeneous model, this is just the average expit of the alpha_0 parameter
            if model == 'homogeneous':
                psi_values = expit(posterior.alpha0)
                node_df['psi_homogeneous'] = psi_values.mean()
                node_df['T_prob_homogeneous'] = node_df['A_prob_homogeneous']*node_df['psi_homogeneous']

            #In the heterogeneous model, we must compute psi for every iteration by doing
            # the linear combination with our covariates:
            else:
                coefficients = posterior[[col for col in cols_params if 'alpha' in col]].values
                psi_values = np.array([expit(row[0] + np.sum(row[1:]*X_multivariate, axis=1)) for row in coefficients])
                node_df['psi_multivariate'] = psi_values.mean(axis=0)
                node_df['T_prob_multivariate'] = node_df['A_prob_multivariate']*node_df['psi_multivariate']

                #We also add the parameter posteriors to the pooling dictionary:
                for param_to_pool, posteriors_to_pool in posteriors_multivariate.items():
                    posteriors_to_pool.append(posterior_df[param_to_pool].values)
                    summary = summary_df.loc[param_to_pool][['mean', '2.5%', '97.5%', 'r_hat']]
                    summary['Model'] = event
                    summaries_multivariate.append(summary.rename(param_to_pool))

        #We can now save the node variables dataframe:
        node_df.to_csv(f'{args.outputs_processed_dir}{args.mode}/node-variables/{unit}-{event}.csv', index=False)
        
    print('  ...pooling')
    #Let's pool the multivariate coefficients:
    for covariate, samples in posteriors_multivariate.items():
        estimate, CI = pool.get_combined_posterior_from_samples(samples,
                                                                prior_mean=args.psi_prior_mean,
                                                                prior_std=args.psi_prior_sd,
                                                                estimate_type='mean',
                                                                CI_alpha=0.95,
                                                                return_pdf=False,
                                                                plot=False)
        sr = pd.Series(['pooled', estimate, CI[0], CI[1], np.nan],
                       index=['Model', 'mean', '2.5%', '97.5%', 'r_hat'],
                       name=covariate)
        summaries_multivariate.append(sr)
        
    #Save the multivariate coefficients:
    summaries_multivariate_df = pd.concat(summaries_multivariate, axis=1)
    summaries_multivariate_df = summaries_multivariate_df.T
    summaries_multivariate_df.index.name = 'Parameter'
    summaries_multivariate_df.to_csv(f'{args.outputs_processed_dir}{args.mode}/pooled-coefficients/{unit}-multivariates.csv')

    #Now let's look at univariates:
    if args.univariates:
        print('  MODEL: univariates')
        posteriors_univariate = {cov:[] for cov in vars._covariates + vars._covariates_unused if cov != 'population'}
        summaries_univariate = []

        #We need to again look at all the storms:
        for event in args.storms:
            outputs_dir = f'{args.outputs_raw_dir}{args.mode}/floods-{event}/'
            files = os.listdir(outputs_dir)

            #And we will consider every covariate in each storm:
            for covariate in posteriors_univariate.keys():
                
                #Let's collect all csv files corresponding to the covariate:
                if covariate != 'population_density_land':
                    file_paths = [f'{outputs_dir}{file}' for file in files if f'-{unit}-{covariate}-' in file]
                #In the case of the population density land for census, just copy the density:
                else:
                    if unit=='census':
                        file_paths = [f'{outputs_dir}{file}' for file in files if f'-{unit}-population_density-' in file]
                
                #The sample dfs correspond to all chains we run in the model.
                sample_dfs = [pd.read_csv(file).drop('Unnamed: 0', axis=1) for file in file_paths if 'sample' in file]
                _ = [df.insert(0, 'chain', k+1) for k, df in enumerate(sample_dfs)]
                
                #After we crop the posterior distributions, we can concatenate all these chains and save:
                posterior_dfs = [df.iloc[args.MCMC_n_burnin:] for df in sample_dfs]
                posterior_df = pd.concat(posterior_dfs, ignore_index=True)
    
                #Summarize in a series:
                summary = posterior_df['alpha1'].describe(percentiles=[0.025, 0.975]).rename(covariate)
                summary['r_hat'] = ev.compute_rhats(posterior_df, parameter_cols=['alpha1'], chain_col='chain')[0]
                summary['Model'] = event
                summaries_univariate.append(summary[['Model', 'mean', '2.5%', '97.5%', 'r_hat']])
    
                #Add to the dictionary:
                posteriors_univariate[covariate].append(posterior_df['alpha1'].values)
                
        #After going through all the covariates and all the storms, we can pool:
        for covariate, samples in posteriors_univariate.items():
            estimate, CI = pool.get_combined_posterior_from_samples(samples,
                                                                    prior_mean=args.psi_prior_mean,
                                                                    prior_std=args.psi_prior_sd,
                                                                    estimate_type='mean',
                                                                    CI_alpha=0.95,
                                                                    return_pdf=False,
                                                                    plot=False)
            sr = pd.Series(['pooled', estimate, CI[0], CI[1], np.nan],
                           index=['Model', 'mean', '2.5%', '97.5%', 'r_hat'],
                           name=covariate)
            summaries_univariate.append(sr)
        
        #Save the summaries:
        summaries_univariate_df = pd.concat(summaries_univariate, axis=1)
        summaries_univariate_df = summaries_univariate_df.T
        summaries_univariate_df.index.name = 'Parameter'
        summaries_univariate_df.to_csv(f'{args.outputs_processed_dir}{args.mode}/pooled-coefficients/{unit}-univariates.csv')
        