import numpy as np
import pandas as pd
from IPython.display import Image

import arviz as az
from tqdm import tqdm

from sklearn.metrics import roc_curve, roc_auc_score, mean_squared_error, recall_score, precision_score
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from scipy.special import expit
from scipy.stats import t

import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'cm'

import os

##################################################################################################################################

import sys
sys.path.append('../d03_src/')
import vars
from evaluation_visualization import plot_pairwise, plot_chains_and_posteriors, plot_calibration, plot_evaluation_scatterplot, plot_AUC, plot_rhatdistributions, plot_classifier_calibration
from utils import latex
from process_demographics import include_covariates
from process_graph import generate_graph_census, generate_graph_geohash
from process_complaints import get_complaints

##################################################################################################################################
#PROCESS EXPERIMENTS:
def add_psi(alphas, ny_gdf, ny_unit='census', covariate_names=['log_population'], covariate_standardize=True):
            
    #Get the covariates:
    covariates = include_covariates(ny_gdf,
                                    gdf_type=ny_unit,
                                    covariates_names=covariate_names,
                                    standardize=False)
    #Get the report rate:
    covariates_std = (covariates - covariates.mean(axis=0))/(covariates.std(axis=0))
    covariates_augmented = np.insert(covariates_std if covariate_standardize else covariates, 0, 1, axis=1)
    psi = expit(np.dot(covariates_augmented, alphas))

    return psi, covariates

def experiments_to_arviz(sampled_params_df,
                         true_params_df=None,
                         n_burnin=0,
                         param_names=['psi', 'theta0', 'theta1']):
    """
    Convert experimental results with either synthetic
      or real data to ArViz Inference Data objects. Each
      experiment can contain multiple chains. All
      dataframes passed must contain columns `run`
      (experiment no.) and `chain`, even if values are
      constant across chains for true parameters.
    """
    #If we didn't pass the number of runs or chains, assume they come from the same:
    if 'chain' not in sampled_params_df.columns: sampled_params_df.insert(0, 'chain', 1)
    if 'run' not in sampled_params_df.columns: sampled_params_df.insert(0, 'run', 1)
    
    if true_params_df is not None:
        if 'chain' not in true_params_df.columns: true_params_df.insert(0, 'chain', 1)
        if 'run' not in true_params_df.columns: true_params_df.insert(0, 'run', 1)
    
    #Get the number of experiments (runs x chains):
    experiments = np.unique(sampled_params_df[['run', 'chain']].values, axis=0)
    if true_params_df is not None: assert (np.unique(true_params_df[['run', 'chain']].values, axis=0) == experiments).all()

    #Infer the network size by groundtruth states (and check with observed data):
    A_cols = [A_col for A_col in sampled_params_df.columns if 'A' in A_col]
    N = len(A_cols)
    assert N > 0
    if true_params_df is not None: T_cols = [T_col for T_col in true_params_df.columns if 'T' in T_col]
    if true_params_df is not None: assert len(T_cols) == N

    #Iterate over all experiments:
    inference_data_list = []
    for experiment in tqdm(np.unique(experiments[:,0])):

        #Find the chains we ran on that experiment:
        chains = experiments[np.in1d(experiments[:,0], experiment)][:,1]
        n_chains = len(chains)

        #Collect the posteriors for each chain:
        full_posterior_dfs = [sampled_params_df[(sampled_params_df.run == experiment) & (sampled_params_df.chain == chain)].drop(['run', 'chain'], axis=1).reset_index(drop=True) for chain in chains]
        n_draws = min(len(posterior_df) for posterior_df in full_posterior_dfs)
        posterior_dfs = [posterior_df[n_burnin:n_draws] for posterior_df in full_posterior_dfs] #account for variance due to thinning fraction

        #Make dictionaries for arviz:
        sampled_param_dict = {param : np.vstack([posterior_dfs[k][param].values for k in range(len(chains))]) for param in param_names}        
        A_dict = {'A': np.array([posterior_dfs[k][A_cols].values for k in range(len(chains))])}
        sampled_param_dict.update(A_dict)

        #We may have passed burn-in iterations too:
        if n_burnin:
            warmup_posterior_dfs = [posterior_df[:n_burnin] for posterior_df in full_posterior_dfs]
            warmup_param_dict = {param : np.vstack([warmup_posterior_dfs[k][param].values for k in range(len(chains))]) for param in param_names}            
            warmup_A_dict = {'A': np.array([warmup_posterior_dfs[k][A_cols].values for k in range(len(chains))])}
            warmup_param_dict.update(warmup_A_dict)

        #Create a posterior predictive distribution:
        if 'psi' in param_names:
            posterior_pred_dict = {'T': np.random.binomial(1, sampled_param_dict['psi'][..., None]*(sampled_param_dict['A']/2 + 0.5))}
        else:
            posterior_pred_dict=None

        #We may have passed observed T (and A, if synthetic):
        if true_params_df is not None:
            true_df = true_params_df[true_params_df.run == experiment].iloc[0]
            obs_data_dict = {'T': true_df[T_cols].values}
            A_dict = {'A': true_df[A_cols].values} if A_cols[0] in true_df else None
            if A_dict is not None: obs_data_dict.update(A_dict)

        #Convert to arviz:
        inference_data = az.from_dict(posterior=sampled_param_dict,
                                      posterior_predictive=posterior_pred_dict,
                                      warmup_posterior=warmup_param_dict if n_burnin else None,
                                      save_warmup=True,
                                      observed_data=obs_data_dict if true_params_df is not None else None,
                                      coords={'nodes': np.arange(N)},
                                      dims={'A':['nodes'], 'T': ['nodes']})

        inference_data_list.append(inference_data)
        
    return inference_data_list

def summarize_params(all_inferred_params,
                     true_params=None,
                     show=True,
                     rhats=None,
                     cols_to_show=['mean', 'std', '2.5%', '50%', '97.5%']):
    
    #This avoids modifying the original dictionaries and excludes the array:
    all_inferred_params_for_df = [{param:chain[param] for param in chain if param != 'A'}
                                  for chain in all_inferred_params]
    
    #In case we passed an A array, we need to extract the mean!
    for chain, chain_for_df in zip(all_inferred_params, all_inferred_params_for_df):
        if 'A' in chain: chain_for_df['A_mean'] = [A_arr.mean()for A_arr in chain['A']]
    
    inferred_params_df = pd.concat([pd.DataFrame(d) for d in all_inferred_params_for_df])
    summarized_df = inferred_params_df.describe([0.025*k for k in range(41)]).transpose()
    
    #Check if we have true parameters:
    if true_params is not None and 'A' in true_params:
        true_params['A_mean'] = np.mean(true_params['A'])
        summarized_df['true'] = pd.Series({p: float(true_params[p]) for p in true_params if p in summarized_df.index})
        if 'true' not in cols_to_show: cols_to_show.append('true')
        
    #Check if we have the rhats:
    if rhats is not None:
        summarized_df['r_hats'] = pd.Series(rhats)
        if 'r_hats' not in cols_to_show: cols_to_show.append('r_hats')
        
    if show: print(summarized_df.round(2)[cols_to_show])
    return summarized_df


def _get_posterior_dicts(sampled_params_df,
                         experiment=1,
                         chains=[1],
                         n_burnin=10_000,
                         param_names=['theta0', 'theta1', 'psi']):

    """
    Return first with burn-in, then without
    """
    #Collect the full posteriors (and post-burnin posteriors) for each chain:
    full_posterior_dfs = [sampled_params_df[(sampled_params_df.run == experiment) & (sampled_params_df.chain == chain)].drop(['run', 'chain'], axis=1).reset_index(drop=True) for chain in chains]
    n_draws = min(len(posterior_df) for posterior_df in full_posterior_dfs) #account for sample size variance due to thinning fraction
    full_posterior_dfs = [posterior_df[:n_draws] for posterior_df in full_posterior_dfs]
    posterior_dfs = [posterior_df[n_burnin:n_draws] for posterior_df in full_posterior_dfs]

    #Turn them into dictionaries (be careful to include A as an array):
    def add_A(p_dict, p_df, not_A=len(param_names)):
        A_cols = [A_col for A_col in p_df.columns if 'A' in A_col]
        p_dict['A'] = p_df.loc[:,A_cols].values/2 + 0.5
        return p_dict

    full_posterior_dicts = [add_A(full_posterior_df[param_names].to_dict(orient='list'), full_posterior_df) for full_posterior_df in full_posterior_dfs]
    posterior_dicts = [add_A(posterior_df[param_names].to_dict(orient='list'), posterior_df) for posterior_df in posterior_dfs]

    return full_posterior_dicts, posterior_dicts

def _add_chain_and_run(sampled_params_df, true_params_df=None):
    
    
    if 'chain' not in sampled_params_df.columns: sampled_params_df.insert(0, 'chain', 1)
    if 'run' not in sampled_params_df.columns: sampled_params_df.insert(0, 'run', 1)
    
    if true_params_df is not None:
        if 'chain' not in true_params_df.columns: true_params_df.insert(0, 'chain', 1)
        if 'run' not in true_params_df.columns: true_params_df.insert(0, 'run', 1)

    return sampled_params_df, true_params_df

def experiments_to_nodestats(sampled_params_df,
                             true_params_df=None,
                             test_T_array=None,
                             covariate_names=['log_population'],
                             covariate_standardize=True,
                             ny_gdf=None,
                             ny_unit='census',
                             n_burnin=10_000,
                             param_names=['theta0', 'theta1', 'psi']):
    """
    Compute node specific statistics:
         - inferred probability of event
         - inferred average report rate
         - observed report
    If available, also record:
         - ground truth of event
         - ground truth of report rate
    """
    #Process if we didn't pass the number of runs or chains:                       
    sampled_params_df, true_params_df = _add_chain_and_run(sampled_params_df,
                                                           true_params_df)
                                 
    #Get the number of experiments (runs x chains):
    experiments = np.unique(sampled_params_df[['run', 'chain']].values, axis=0)
    if true_params_df is not None: assert (np.unique(true_params_df[['run', 'chain']].values, axis=0) == experiments).all()
        
    #Infer the network size by groundtruth states (and check with observed data):
    A_cols = [A_col for A_col in sampled_params_df.columns if 'A' in A_col]
    N = len(A_cols)
    assert N > 0
    if true_params_df is not None: T_cols = [T_col for T_col in true_params_df.columns if 'T' in T_col]
    if true_params_df is not None: assert len(T_cols) == N
    if true_params_df is not None: only_true_T = False if param_names[0] in true_params_df.columns else True

    nodestats_list = []
    for experiment in tqdm(np.unique(experiments[:,0])):

        #Find the chains we ran on that experiment:
        chains = experiments[np.in1d(experiments[:,0], experiment)][:,1]
        n_chains = len(chains)
        
        #Collect the posteriors:
        _, posterior_dicts = _get_posterior_dicts(sampled_params_df,
                                                  experiment, chains,
                                                  n_burnin, param_names)
        #Merge all posteriors:
        full_posterior = {param:[] for param in (param_names+['A'] if 'A' not in param_names else param_names)}
        for posterior in posterior_dicts:
            for param in full_posterior.keys():
                full_posterior[param].extend(posterior[param])

        #Get inferred A probability per node:
        A_prob = np.mean(np.vstack(full_posterior['A']),axis=0)

        #Create the dataframe:
        nodestats_df = pd.DataFrame(A_prob, columns=['inf_Aprob'])
        nodestats_df.index.name='node'

        #Add the inferred report rate:
        if 'psi' in param_names:
            nodestats_df['inf_psi'] = np.mean(full_posterior['psi'])
            
        if 'alpha0' in param_names:
            alpha_columns = [param for param in param_names if 'alpha' in param]
            alpha_columns.sort(key=lambda x: int(x[5:]))
            alphas = [np.mean(full_posterior[alpha]) for alpha in alpha_columns]

            psi, covariates = add_psi(alphas, ny_gdf, ny_unit, covariate_names, covariate_standardize)
            nodestats_df[covariate_names] = covariates
            nodestats_df['inf_psi'] = psi

        #Collect the true parameters:
        if true_params_df is not None:
            true_params_exp = true_params_df[true_params_df.run == experiment].iloc[0]
            true_params_dict = dict(true_params_exp[param_names]) if not only_true_T else dict()
            true_params_dict['T'] = true_params_exp[T_cols].values
            nodestats_df['T_train'] = true_params_dict['T']
            if test_T_array is not None:
                nodestats_df['T_test'] = test_T_array
                nodestats_df['T_full'] = (nodestats_df['T_train'] + nodestats_df['T_test']).clip(upper=1).astype(int)

            #Check if we also know ground truth (synth data)
            true_A_cols = [A_col for A_col in true_params_exp.index if 'A' in A_col]
            if len(true_A_cols) > 0:
                true_params_dict['A'] = true_params_exp[true_A_cols].values/2 + 0.5
                nodestats_df['A'] = true_params_dict['A']
            if 'psi' in true_params_dict:
                nodestats_df['psi'] = true_params_dict['psi']

        #Add to list
        nodestats_list.append(nodestats_df)
            
    return nodestats_list

def experiments_to_summaries(sampled_params_df,
                             true_params_df=None,
                             arviz_inference_list=None,
                             n_burnin=5000,
                             plot=True,
                             show=True,
                             save_plots_dir='plots/',
                             save_plots_name='exp',
                             save_plots_base_idx=0,
                             param_names=['psi', 'theta0', 'theta1'],
                             ranges={'theta0': (-0.5, 0.5), 'theta1':(-0.1, 0.3)}):
    
    #Process if we didn't pass the number of runs or chains:                       
    sampled_params_df, true_params_df = _add_chain_and_run(sampled_params_df,
                                                           true_params_df)
    
    #Get the number of experiments (runs x chains):
    experiments = np.unique(sampled_params_df[['run', 'chain']].values, axis=0)
    if true_params_df is not None: assert (np.unique(true_params_df[['run', 'chain']].values, axis=0) == experiments).all()

    #Infer the network size by groundtruth states (and check with observed data):
    A_cols = [A_col for A_col in sampled_params_df.columns if 'A' in A_col]
    N = len(A_cols)
    assert N > 0
    if true_params_df is not None: T_cols = [T_col for T_col in true_params_df.columns if 'T' in T_col]
    if true_params_df is not None: assert len(T_cols) == N
    if true_params_df is not None: only_true_T = False if param_names[0] in true_params_df.columns else True

    #Iterate over all experiments:
    summary_list = []
    for experiment in tqdm(np.unique(experiments[:,0])):

        #Find the chains we ran on that experiment:
        chains = experiments[np.in1d(experiments[:,0], experiment)][:,1]
        n_chains = len(chains)

        #Collect the full posteriors (and post-burnin posteriors) for each chain:
        full_posterior_dicts, posterior_dicts = _get_posterior_dicts(sampled_params_df,
                                                                     experiment, chains,
                                                                     n_burnin, param_names)

        #Get the r_hats if we have an arviz:
        if arviz_inference_list is not None:
            rhat_xarray = az.rhat(arviz_inference_list[experiment-1], var_names=param_names)
            rhats = {param: float(rhat_xarray[param].values) for param in param_names}

        #Collect the true parameters:
        if true_params_df is not None:
            true_params_exp = true_params_df[true_params_df.run == experiment].iloc[0]
            true_params_dict = dict(true_params_exp[param_names]) if not only_true_T else dict()
            A_cols = [A_col for A_col in true_params_exp.index if 'A' in A_col]
            if len(A_cols) > 0: true_params_dict['A'] = true_params_exp[A_cols].values/2 + 0.5
            true_params_dict['T'] = true_params_exp[T_cols].values

        #Plot if we want to:
        if plot:
            fig, ax = plot_chains_and_posteriors(full_posterior_dicts,
                                                 true_params=true_params_dict if true_params_df is not None else None,
                                                 burnin=n_burnin,
                                                 passed_T=True if true_params_df is not None else False,
                                                 rhats=rhats if arviz_inference_list is not None else None,
                                                 ranges=ranges,
                                                 param_names=param_names)
            if save_plots_dir: fig.savefig(f'{save_plots_dir}_{save_plots_name}_{save_plots_base_idx+experiment}.png')
            if show: plt.show()
            Ax = plot_pairwise(posterior_dicts)

        #Summarize the parameters:
        summary_df = summarize_params(posterior_dicts,
                                      true_params_dict if true_params_df is not None else None,
                                      rhats=rhats if arviz_inference_list is not None else None,
                                      show=True)
        summary_list.append(summary_df)
        
    return summary_list
                                 
##################################################################################################################################
#CALIBRATION

def evaluate_summaries(summaries,
                       param_names=['theta0', 'psi'],
                       confidence=0.95,
                       verbose=True):
    
    #Collect the endpoints of the interval as a df column name
    def get_pct_str(val):
        pct_str = f'{val:.0%}' if 100*val%1 == 0 else f'{val:.1%}'
        return pct_str
    
    lower_bound = get_pct_str((1-confidence)/2)
    upper_bound = get_pct_str((1+confidence)/2)
    
    means = {param:[] for param in param_names}
    medians = {param:[] for param in param_names}
    lower = {param:[] for param in param_names}
    upper = {param:[] for param in param_names}
    true = {param:[] for param in param_names}
    is_within_interval = {param:[] for param in param_names}

    for summary_df in summaries:
        for param in param_names:
            means[param].append(summary_df.loc[param, 'mean'])
            medians[param].append(summary_df.loc[param, '50%'])
            lower[param].append(summary_df.loc[param, lower_bound])
            upper[param].append(summary_df.loc[param, upper_bound])
            true[param].append(summary_df.loc[param, 'true'])

            is_within_interval[param].append(True if (summary_df.loc[param, 'true'] <= summary_df.loc[param, upper_bound]
                                                      and summary_df.loc[param, 'true']>=summary_df.loc[param, lower_bound])
                                             else False)
    if verbose:
        print(f'Percentage of the time true parameter is in the {confidence:.1%}% interval of the posterior:')     
        for param in param_names:
            print(f"{param}: {np.array(is_within_interval[param]).mean():.1%}")
        
    #Create a dictionary of dataframes to return (they will be used for plotting):
    df_dict = {param:pd.DataFrame([true[param], means[param], lower[param], upper[param]],
                                  index=['True', 'Estimate', 'Lower bound', 'Upper bound']).T
               for param in param_names}
    return df_dict

def get_estimate_sd_and_CI(summary, param, estimate='mean', confidence_level=0.95):
    
    param_row = summary.loc[param]

    if estimate == 'median': estimate = '50%'
    param_estimate = param_row[estimate]
    param_sd = param_row['std']

    lower_bound = str(round(50*(1 - confidence_level),1))
    upper_bound = str(round(50*(1 + confidence_level),1))
    param_LB = param_row[f"{lower_bound[:-2] if lower_bound[-1]=='0' else lower_bound}%"]
    param_UB = param_row[f"{upper_bound[:-2] if upper_bound[-1]=='0' else upper_bound}%"]
    param_CI = (param_LB, param_UB)

    return param_estimate, param_sd, param_CI
    
#UNIVARIATE SUMMARIES:
def print_CIs_table(univariate_summary_files, job_dir, est='mean', round=3, cov_map=vars._covariates_names):
    all_covariates = []
    for summary_file in univariate_summary_files:
    
        cov_name = summary_file.split('.')[0].split('-')[-1]
        summary = pd.read_csv(f'{job_dir}summaries/{summary_file}').rename({'Unnamed: 0':'parameter'}, axis=1).set_index('parameter')
        slope = summary.loc['alpha1'].rename(cov_name)
        all_covariates.append(slope)
    
    all_covariates_summary = pd.concat(all_covariates,axis=1).T
    medians = all_covariates_summary.loc[:,est].rename('estimate')
    UB = all_covariates_summary.loc[:,'97.5%'].rename('upper bound')
    LB = all_covariates_summary.loc[:,'2.5%'].rename('lower bound')
    
    CIs_table = pd.DataFrame([], index=list(cov_map.keys()))
    CIs_table['Estimate'] = medians
    CIs_table['Lower bound'] = LB
    CIs_table['Upper bound'] = UB
    
    CIs_table.index = list(cov_map.values())
    CIs_table = CIs_table.round(round).replace(np.nan, '-')

    return CIs_table
##################################################################################################################################
#CONVERGENCE

def get_rhats(summaries,
              param_names=['theta0', 'theta1', 'psi'],
              plot=True):
    rhats = {}
    for param in param_names:
      rhats[param] = [summary.loc[param].r_hats for summary in summaries]

    if plot: _ = plot_rhatdistributions(rhats)

    return rhats

##################################################################################################################################
#Classifier Evaluation Metrics

def get_cols_for_evaluation(real_data=False,
                            non_observed_only=True, use_Aprob=True,
                            restrict_to_Ttrain_zero=None):
        
    #Depending on whether we are using calibration vs inference,
    # Pr(T) or Pr(A), adjust the columns:
    reports_col = 'T_train'
    classes_col = 'A' if not real_data else ('T_full' if (real_data and not non_observed_only) else 'T_test')
    scores_col = 'inf_Aprob' if (use_Aprob or not real_data) else 'inf_Tprob'

    #Decide whether to filter to unreported nodes:
    if restrict_to_Ttrain_zero is None: restrict_to_Ttrain_zero=False if classes_col=='T_test' else True

    return classes_col, scores_col, reports_col, restrict_to_Ttrain_zero

def compute_RMSE_and_AUC_with_CIs(nodestats_df,
                                  real_data=False,
                                  non_observed_only=True, use_Aprob=True, restrict_to_Ttrain_zero=None,
                                  confidence_level=0.95,
                                  n_parameters=2,
                                  n_bootstrap=1000,
                                  n_bootstrap_eff=True):
    
    #Get cols:
    classes_col, scores_col, reports_col, filter = get_cols_for_evaluation(real_data,
                                                                           non_observed_only,
                                                                           use_Aprob,
                                                                           restrict_to_Ttrain_zero)
    if filter: nodestats_df = nodestats_df[nodestats_df[reports_col]==0]
    classes = nodestats_df[classes_col].values
    scores = nodestats_df[scores_col].values
    N = len(classes)

    #Compute estimates:
    RMSE = mean_squared_error(classes, scores, squared=False)
    AUC = roc_auc_score(classes, scores)

    #Compute RMSE CI:
    dof = N - n_parameters
    critical_value = t.ppf((1+confidence_level)/2, df=dof)
    RMSE_CI_width = critical_value * RMSE/np.sqrt(N)
    RMSE_CI = (RMSE-RMSE_CI_width, RMSE+RMSE_CI_width)

    #Compute AUC CI:
    AUC_values = []
    while len(AUC_values) != n_bootstrap:
        #Get a bootstraped sample:
        boot_idx = resample(range(N), replace=True)
        boot_classes = classes[boot_idx]
        boot_scores = scores[boot_idx]
        
        # Calculate AUC for the bootstrap sample
        if len(set(boot_classes)) != 1:
            auc = roc_auc_score(boot_classes, boot_scores)
            AUC_values.append(auc)
        #In case we don't have a split test set, we only count
        # the iteration if we dont care about the effective
        # sample size:
        else:
            if not n_bootstrap_eff: AUC_values.append(np.nan)
    #Compute the CI ignoring nan values
    AUC_CI = np.nanpercentile(AUC_values,
                              [50*(1-confidence_level), 50*(1+confidence_level)])
    
    return RMSE, AUC, RMSE_CI, tuple(AUC_CI)

def compute_CI_differences_RMSE_AUC_topkrecall(nodestats_df_list,
                                               baseline_df,
                                               real_data=False,
                                               non_observed_only=True, use_Aprob=True, restrict_to_Ttrain_zero=None,
                                               confidence_level=0.95,
                                               n_bootstrap=1000,
                                               n_bootstrap_eff=True,
                                               recall_k=100,
                                               precision=3):
    #Get cols:
    classes_col, scores_col, reports_col, filter = get_cols_for_evaluation(real_data,
                                                                           non_observed_only,
                                                                           use_Aprob,
                                                                           restrict_to_Ttrain_zero)
    if filter:
        nodestats_df_list = [df[df[reports_col]==0] for df in nodestats_df_list]
        baseline_df = baseline_df[baseline_df[reports_col]==0]
        
    classes = baseline_df[classes_col].values
    baseline_scores = baseline_df[scores_col].values
    N = len(classes)
    scores_list = [df[scores_col].values for df in nodestats_df_list]

    #RMSE:
    rmse_baseline = mean_squared_error(classes, baseline_scores, squared=False)
    RMSE = [mean_squared_error(classes, scores, squared=False) for scores in scores_list] + [rmse_baseline]
    RMSE_base = [rmse - rmse_baseline for rmse in RMSE]
    RMSE_relative = RMSE[0]-RMSE[1]
                                                   
    #AUC                                        
    auc_baseline = roc_auc_score(classes, baseline_scores)
    AUC = [roc_auc_score(classes, scores) for scores in scores_list] + [auc_baseline]
    AUC_base = [auc - auc_baseline for auc in AUC]
    AUC_relative = AUC[0]-AUC[1]

    #top-K recall:
    def to_binary(scores, K=recall_k):
        top_K_idx = np.argpartition(scores, -K)[-K:]
        binary_scores = np.zeros_like(scores)
        binary_scores[top_K_idx] = 1
        return binary_scores
    
    recall_baseline = recall_score(classes, to_binary(baseline_scores), zero_division=np.nan)
    RECALL = [recall_score(classes, to_binary(scores), zero_division=np.nan) for scores in scores_list] + [recall_baseline]
    RECALL_base = [recall - recall_baseline for recall in RECALL]
    RECALL_relative = RECALL[0]-RECALL[1]  
                                                   
    #Bootstrap:
    RMSE_list = []                                   
    RMSE_base_improvement_list = []
    RMSE_base_improvement_pval_list = []
    RMSE_relative_12_improvement_list = []
    RMSE_relative_12_improvement_pval_list = []

    AUC_list = []
    AUC_base_improvement_list = []
    AUC_base_improvement_pval_list = []
    AUC_relative_12_improvement_list = []
    AUC_relative_12_improvement_pval_list = []
                                                   
    RECALL_list = []
    RECALL_base_improvement_list = []
    RECALL_base_improvement_pval_list = []
    RECALL_relative_12_improvement_list = []
    RECALL_relative_12_improvement_pval_list = []
                                          
    while len(AUC_base_improvement_list) != n_bootstrap:
        
        #Get a bootstraped sample:
        boot_idx = resample(range(N), replace=True)
        boot_classes = classes[boot_idx]
        boot_baseline_scores = baseline_scores[boot_idx]
        boot_scores_list = [scores[boot_idx] for scores in scores_list]
        
        # Calculate AUC and RMSE for the bootstrap sample
        if len(set(boot_classes)) != 1:

            #RMSE:
            baseline_rmse = mean_squared_error(boot_classes, boot_baseline_scores, squared=False)
            rmse_list = [mean_squared_error(boot_classes, boot_scores, squared=False) for boot_scores in boot_scores_list]
            RMSE_list.append(rmse_list+[baseline_rmse])
            RMSE_base_improvement_list.append([rmse-baseline_rmse for rmse in rmse_list])
            RMSE_relative_12_improvement_list.append(rmse_list[0]-rmse_list[1])
            RMSE_base_improvement_pval_list.append([rmse > baseline_rmse for rmse in rmse_list])
            RMSE_relative_12_improvement_pval_list.append(int(rmse_list[0]>rmse_list[1]))

            #AUC:
            baseline_auc = roc_auc_score(boot_classes, boot_baseline_scores)
            auc_list = [roc_auc_score(boot_classes, boot_scores) for boot_scores in boot_scores_list]
            AUC_list.append(auc_list+[baseline_auc])
            AUC_base_improvement_list.append([auc-baseline_auc for auc in auc_list])
            AUC_relative_12_improvement_list.append(auc_list[0]-auc_list[1])
            AUC_base_improvement_pval_list.append([int(auc > baseline_auc) for auc in auc_list])
            AUC_relative_12_improvement_pval_list.append(int(auc_list[0]>auc_list[1]))

            #top-K recall:
            baseline_recall = recall_score(boot_classes, to_binary(boot_baseline_scores), zero_division=np.nan)
            recall_list = [recall_score(boot_classes, to_binary(boot_scores), zero_division=np.nan) for boot_scores in boot_scores_list]
            RECALL_list.append(recall_list+[baseline_recall])
            RECALL_base_improvement_list.append([recall-baseline_recall for recall in recall_list])
            RECALL_relative_12_improvement_list.append(recall_list[0]-recall_list[1])
            RECALL_base_improvement_pval_list.append([recall > baseline_recall for recall in recall_list])
            RECALL_relative_12_improvement_pval_list.append(int(recall_list[0]>recall_list[1]))
            
        #In case we don't have a split test set, we only count
        # the iteration if we dont care about the effective
        # sample size:
        elif not n_bootstrap_eff:
            RMSE_base_improvement_list.append([np.nan]*N_models)
            RMSE_relative_12_improvement_list.append(np.nan)
            RMSE_list.append(np.nan)
            
            AUC_base_improvement_list.append([np.nan]*N_models)
            AUC_relative_12_improvement_list.append(np.nan)
            AUC_list.append(np.nan)

            RECALL_base_improvement_list.append([np.nan]*N_models)
            RECALL_relative_12_improvement_list.append(np.nan)
            RECALL_list.append(np.nan)
            
    #Compute the CI ignoring nan values
    q = [50*(1-confidence_level), 50*(1+confidence_level)]
                                            
    RMSE_CI = np.round(np.nanpercentile(np.array(RMSE_list), q, axis=0),precision)
    RMSE_CI_base = np.round(np.nanpercentile(np.array(RMSE_base_improvement_list), q, axis=0),precision)
    RMSE_CI_relative = np.round(np.nanpercentile(np.array(RMSE_relative_12_improvement_list), q, axis=0),precision)                                            
    RMSE_base_pval = np.mean(np.array(RMSE_base_improvement_pval_list), axis=0)
    RMSE_relative_pval = np.mean(RMSE_relative_12_improvement_pval_list)
                                                   
    AUC_CI = np.round(np.nanpercentile(np.array(AUC_list), q, axis=0),precision)                                          
    AUC_CI_base = np.round(np.nanpercentile(np.array(AUC_base_improvement_list), q, axis=0),precision)
    AUC_CI_relative = np.round(np.nanpercentile(np.array(AUC_relative_12_improvement_list), q, axis=0),precision)
    AUC_base_pval = 1 - np.mean(np.array(AUC_base_improvement_pval_list), axis=0)
    AUC_relative_pval = 1 - np.mean(AUC_relative_12_improvement_pval_list)
                                                   
    RECALL_CI = np.round(np.nanpercentile(np.array(RECALL_list), q, axis=0),precision)                                          
    RECALL_CI_base = np.round(np.nanpercentile(np.array(RECALL_base_improvement_list), q, axis=0),precision)
    RECALL_CI_relative = np.round(np.nanpercentile(np.array(RECALL_relative_12_improvement_list), q, axis=0),precision)
    RECALL_base_pval = 1 - np.mean(np.array(RECALL_base_improvement_pval_list), axis=0)
    RECALL_relative_pval = 1 - np.mean(RECALL_relative_12_improvement_pval_list)

    RMSE_returns = [(RMSE, RMSE_CI), (RMSE_base, RMSE_CI_base, RMSE_base_pval), (RMSE_relative, RMSE_CI_relative, RMSE_relative_pval)]
    AUC_returns = [(AUC, AUC_CI), (AUC_base, AUC_CI_base, AUC_base_pval), (AUC_relative, AUC_CI_relative, AUC_relative_pval)]
    RECALL_returns = [(RECALL, RECALL_CI), (RECALL_base, RECALL_CI_base, RECALL_base_pval), (RECALL_relative, RECALL_CI_relative, RECALL_relative_pval)]
                                            
    return RMSE_returns, AUC_returns, RECALL_returns

def compute_recall(nodestats,
                   tresh=0.5, top_k=False, recall_k=100,
                   real_data=False,
                   non_observed_only=True, use_Aprob=True,
                   restrict_to_Ttrain_zero=None):
    #top-K recall:
    def to_binary(scores, K=recall_k):
        top_K_idx = np.argpartition(scores, -K)[-K:]
        binary_scores = np.zeros_like(scores)
        binary_scores[top_K_idx] = 1
        return binary_scores
        
    #Ensure we have a list of dataframes for multiple experiments:
    if type(nodestats) == pd.DataFrame: nodestats = [nodestats]
        
    #Get cols:
    classes_col, scores_col, reports_col, filter = get_cols_for_evaluation(real_data,
                                                                           non_observed_only,
                                                                           use_Aprob,
                                                                           restrict_to_Ttrain_zero)

    #Get the recall scores:
    recall_list = []
    for df in nodestats:
        if filter: df = df[df[reports_col] == 0]
        scores = np.where(df[scores_col].values > tresh, 1, 0) if not top_k else to_binary(df[scores_col].values, K=recall_k)
        recall = recall_score(df[classes_col].values, scores,
                              zero_division=np.nan)
        recall_list.append(recall)

    return recall_list if len(recall_list)>1 else recall_list[0]

def compute_precision(nodestats,
                      tresh=0.5,
                      real_data=False,
                      non_observed_only=True, use_Aprob=True,
                      restrict_to_Ttrain_zero=None):
                    
    #Ensure we have a list of dataframes for multiple experiments:
    if type(nodestats) == pd.DataFrame: nodestats = [nodestats]
        
    #Get cols:
    classes_col, scores_col, reports_col, filter = get_cols_for_evaluation(real_data,
                                                                           non_observed_only,
                                                                           use_Aprob,
                                                                           restrict_to_Ttrain_zero)

    #Get the precision scores:
    precision_list = []
    for df in nodestats:
        if filter: df = df[df[reports_col] == 0]
        precision = precision_score(df[classes_col].values,
                                    np.where(df[scores_col].values > tresh, 1, 0),
                                    zero_division=np.nan)
        precision_list.append(precision)

    return precision_list if len(precision_list)>1 else precision_list[0]
                     
def compute_RMSE(nodestats,
                 real_data=False,
                 non_observed_only=True, use_Aprob=True,
                 restrict_to_Ttrain_zero=None,
                 CI=False,
                 confidence_level=0.95,
                 params=9):
                    
    #Ensure we have a list of dataframes for multiple experiments:
    if type(nodestats) == pd.DataFrame: nodestats = [nodestats]
        
    #Get cols:
    classes_col, scores_col, reports_col, filter = get_cols_for_evaluation(real_data,
                                                                           non_observed_only,
                                                                           use_Aprob,
                                                                           restrict_to_Ttrain_zero)

    #Get the RMSE:
    RMSE_list = []
    if CI: CI_list = []
    for df in nodestats:
        if filter: df = df[df[reports_col] == 0]
        RMSE = mean_squared_error(df[classes_col].values,
                                  df[scores_col].values,
                                  squared=False)
        RMSE_list.append(RMSE)
        
        #Compute the CI:
        if CI:
            degrees_of_freedom = len(df[classes_col].values) - params
            critical_value = t.ppf((1 + confidence_level)/2, df=degrees_of_freedom)
            margin_of_error = critical_value * (RMSE/np.sqrt(len(df[classes_col].values)))
            CI_list.append((RMSE-margin_of_error, RMSE+margin_of_error))

    if CI:
        return (RMSE_list, CI_list) if len(RMSE_list)>1 else (RMSE_list[0], CI_list[0])
    else:
        return RMSE_list if len(RMSE_list)>1 else RMSE_list[0]

def compute_calibration(nodestats,
                        real_data=False,
                        non_observed_only=True, use_Aprob=True,
                        restrict_to_Ttrain_zero=None,
                        labels=None, n_bins=5,
                        plot=True, show=True, ax=None):
                            
    #Ensure we have a list of dataframes for multiple experiments:
    if type(nodestats) == pd.DataFrame: nodestats = [nodestats]
        
    #Get cols:
    classes_col, scores_col, reports_col, filter = get_cols_for_evaluation(real_data,
                                                                           non_observed_only,
                                                                           use_Aprob,
                                                                           restrict_to_Ttrain_zero)
                            
    #Compute the calibration for each dataframe:
    true_prob_list, pred_prob_list = [], []
    for df in nodestats:
        if filter: df = df[df[reports_col] == 0]
        true_prob, pred_prob = calibration_curve(df[classes_col].values,
                                                 df[scores_col].values,
                                                 n_bins=n_bins)
        #Add to the lists:
        true_prob_list.append(true_prob)
        pred_prob_list.append(pred_prob)
        
    #If we want, plot calibration:
    if plot:
        ax = plot_classifier_calibration(true_prob_list,
                                         pred_prob_list,
                                         labels=labels, ax=ax)
        if show: plt.show()
        
    return (true_prob, pred_prob), ax

def get_AUC_and_rates(df, scores_col, classes_col, filter=False, reports_col='T'):

    #Filter out trivial groundtruth:
    if filter: df = df[df[reports_col] == 0]
    
    #Compute scores and classes:
    scores  = df[scores_col ].values
    classes = df[classes_col].values

    #Ensure the AUC can be computed:
    if len(set(classes)) > 1:
        AUC = roc_auc_score(classes, scores)
        fpr, tpr, _ = roc_curve(classes, scores, pos_label=1)
    else:
        AUC, fpr, tpr = np.nan, np.nan, np.nan
    
    return AUC, fpr, tpr

def compute_AUC(nodestats,
                real_data=False,
                non_observed_only=True, use_Aprob=True,
                restrict_to_Ttrain_zero=None,
                average_AUCs=True, labels=None,
                plot=True, show=True, ax=None):

    #Ensure we have a list of dataframes for multiple experiments:
    if type(nodestats) == pd.DataFrame: nodestats = [nodestats]
        
    #Get cols:
    classes_col, scores_col, reports_col, filter = get_cols_for_evaluation(real_data,
                                                                           non_observed_only,
                                                                           use_Aprob,
                                                                           restrict_to_Ttrain_zero)
    #Compute the AUC for each dataframe:
    AUC_list, fpr_list, tpr_list = [], [], []
    for nodestats_df in nodestats:
        AUC, fpr, tpr = get_AUC_and_rates(df=nodestats_df,
                                          scores_col=scores_col,
                                          classes_col=classes_col,
                                          reports_col=reports_col,
                                          filter=filter)
        #Add to the lists:
        AUC_list.append(AUC)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        
    #If we want, plot AUCs:
    if plot:
        ax = plot_AUC(fpr_list, tpr_list, AUC_list,
                      average_AUCs=average_AUCs, labels=labels, ax=ax)
        if show: plt.show()
        return (AUC_list, ax) if len(AUC_list)>1 else (AUC_list[0], ax)
    else:
        return AUC_list if len(AUC_list)>1 else AUC_list[0]

##################################################################################################################################
#GET FILES

def get_summary_files_inference(job_name, job_dir,
                                verify_T_test=False, train_duration=4, ny_gdf=None, unit='census',
                                include_demographics=False, demographics=vars._covariates+vars._covariates_unused+vars._demographic_categories):
    
    #Make the directories:
    plots_dir = f'{job_dir}plots/'
    summaries_dir = f'{job_dir}summaries/'
    nodestats_dir = f'{job_dir}nodestats/'

    #Add geohash:
    if unit == 'geohash':
        job_name = 'geohash-'+job_name
        
    #Collect possible files:
    plot_file = [file for file in os.listdir(plots_dir) if job_name in file]
    summary_file = [file for file in os.listdir(summaries_dir) if job_name in file]
    nodestats_file = [file for file in os.listdir(nodestats_dir) if job_name in file]
    
    #Verify only one job with the given name:
    assert len(plot_file) == 1
    assert len(summary_file) == 1
    assert len(nodestats_file) == 1
    
    #Read in their format:
    plot = Image(filename=plots_dir+plot_file[0])
    summary = pd.read_csv(summaries_dir+summary_file[0]).rename({'Unnamed: 0':'parameter'}, axis=1).set_index('parameter')
    nodestats = pd.read_csv(nodestats_dir+nodestats_file[0]).set_index('node')

    #Include the test column:
    if verify_T_test:
        if ny_gdf is None:
            if unit == 'census': ny_gdf, _, _ = generate_graph_census(remove_long_edges=True, remove_zeropop=True)
            if unit == 'geohash': ny_gdf, _ = generate_graph_geohash()
                
        train_T, test_T = get_complaints(ny_gdf.to_crs('EPSG:2263'),
                                         event_date=vars._floods_start_date['ida'],
                                         event_duration=7, train_duration=train_duration,
                                         return_test_complaints=True, path_311=vars._path_311_floods)
        
        assert np.all(nodestats.T_train.values == train_T)
        if 'T_test' not in nodestats.columns: nodestats['T_test'] = test_T

    #Include inferred T probability:
    if 'inf_Tprob' not in nodestats.columns: nodestats['inf_Tprob']=nodestats['inf_Aprob']*nodestats['inf_psi']

    #Include demographics:
    if include_demographics:
        if ny_gdf is None:
            if unit == 'census': ny_gdf, _, _ = generate_graph_census(remove_long_edges=True, remove_zeropop=True)
            if unit == 'geohash': ny_gdf, _ = generate_graph_geohash()
        demographic_arr = include_covariates(ny_gdf,
                                             col_to_merge_on='GEOID' if unit=='census' else 'geohash',
                                             covariates_names=demographics, standardize=False)
        nodestats[demographics] = demographic_arr

    return plot, summary, nodestats

def get_summary_files_calibration(job_name, job_dir,
                                  ny_gdf=None, unit='census',
                                  include_demographics=False,
                                  demographics=vars._covariates+vars._covariates_unused+vars._demographic_categories,
                                  print_exp_idx=False,
                                  selected_idx=None):
    #Add geohash:
    if unit == 'geohash':
        job_name = 'geohash-'+job_name
        
    #Make the directories:
    plots_dir = f'{job_dir}{job_name}/plots/'
    summaries_dir = f'{job_dir}{job_name}/summaries/'
    nodestats_dir = f'{job_dir}{job_name}/nodestats/'

    #Collect possible files:
    plot_files = [file for file in os.listdir(plots_dir) if '.png' in file]
    summary_files = [file for file in os.listdir(summaries_dir) if '.csv' in file]
    nodestats_files = [file for file in os.listdir(nodestats_dir) if '.csv' in file]

    def _idx(f):
        fname = f.split('.')[0]
        fidx = fname.split('-')[1] if f[0] != '_' else fname[2:]
        return int(fidx)
        
    #Filter:
    if selected_idx is not None:
        plot_files = [file for file in plot_files if _idx(file) in selected_idx]
        summary_files = [file for file in summary_files if _idx(file) in selected_idx]
        nodestats_files = [file for file in nodestats_files if _idx(file) in selected_idx]
        
                                      
    #Verify jobs have the same length:
    assert len(plot_files) == len(summary_files) == len(nodestats_files)
    
    #Read in their format:
    plot_list = [Image(filename=f'{plots_dir}{file}') for file in plot_files]
    summary_list = [pd.read_csv(f'{summaries_dir}{file}').rename({'Unnamed: 0':'parameter'}, axis=1).set_index('parameter') for file in summary_files]
    nodestats_list = [pd.read_csv(f'{nodestats_dir}{file}').set_index('node') for file in nodestats_files]

    if print_exp_idx:
        exp_idx = [_idx(f) for f in summary_files]
        exp_idx.sort()
        print('Available experiments:')
        print(exp_idx)
        
    #Polish the nodestats files:
    for nodestats in nodestats_list:
        
        #Include inferred T probability:
        if 'inf_Tprob' not in nodestats.columns: nodestats['inf_Tprob']=nodestats['inf_Aprob']*nodestats['inf_psi']

        #Include demographics:
        if include_demographics:
            if ny_gdf is None:
                if unit == 'census': ny_gdf, _, _ = generate_graph_census(remove_long_edges=True, remove_zeropop=True)
                if unit == 'geohash': ny_gdf, _ = generate_graph_geohash()
            if 'GEOID' in ny_gdf:
                ny_gdf.GEOID = ny_gdf.GEOID.apply(lambda x: str(x))
                demographics = [x for x in demographics if x != 'population_density_land']
            demographic_arr = include_covariates(ny_gdf,
                                                 col_to_merge_on='GEOID' if unit=='census' else 'geohash',
                                                 covariates_names=demographics, standardize=False)
            nodestats[demographics] = demographic_arr

    return plot_list, summary_list, nodestats_list

def get_summary_files(job_name, job_dir,
                      verify_T_test=False, train_duration=4, ny_gdf=None,
                      include_demographics=False, demographics=vars._covariates+vars._covariates_unused+vars._demographic_categories,
                      mode='inference', print_exp_idx=False, selected_idx=None, unit='census'):

    if mode == 'inference':
        out = get_summary_files_inference(job_name, job_dir,
                                          verify_T_test, train_duration, ny_gdf, unit,
                                          include_demographics, demographics,)
    elif mode == 'calibration':
        out = get_summary_files_calibration(job_name, job_dir,
                                            ny_gdf, unit,
                                            include_demographics, demographics,
                                            print_exp_idx,
                                            selected_idx=selected_idx)
        
    return out
