##########################################################################################################
#This file contains functions to evaluate the performance of our models
##########################################################################################################

import numpy as np
import pandas as pd
import arviz as az

from sklearn.metrics import roc_curve, roc_auc_score, mean_squared_error, recall_score, precision_score
from sklearn.utils import resample
from scipy.stats import percentileofscore

##########################################################################################################

def compute_rhats(posterior_df, parameter_cols, chain_col='chain'):
    """
    Given a dataframe with posterior distributions for parameters,
        compute the r_hat convergence statistics.
    
    Parameters
    ----------
    posterior_df : pd.DataFrame
        df containing the parameters and a column that identify
        the chain from which the sample came from

    parameter_cols : list
        columns of posterior_df containing parameter info

    chain_col : str
        column of posterior_df containing the chain indicator
        
    Returns
    ----------
    list
        r_hat values, ordered according to parameter_cols
    """
    
    #We can compute the r_hat convergence statistic with arviz, but we need to
    #  have all chains with the same length
    arviz_n_samples = min(posterior_df[chain_col].value_counts())
    
    #Now we make a dictionary of parameters:
    parameters_dict = {param:[] for param in parameter_cols}
    for chain, df in posterior_df[parameter_cols+[chain_col]].groupby(chain_col):
        for param in parameter_cols:
            parameters_dict[param].append(df[param].to_list()[:arviz_n_samples])
            
    #We compute the r_hat statistics:
    r_hats = az.rhat(parameters_dict).to_array().values

    return r_hats

##########################################################################################################

def evaluate_performance(scores,
                         classes,
                         metrics=['AUC', 'RMSE', 'recall'],
                         recall_k=100,
                         confidence_level=0.95,
                         n_bootstrap=1_000,
                         return_improvement=True,
                         all_pairs=False,
                         model_names=None,
                         pval_method='one-sided'):
    """
    Evaluate the performance of several models on the same test
        data (reports or events), according to one or more metrics
        (AUC, RMSE, and recall). Confidence intervals for
        the point estimates are obtained through bootstrapping. If
        requested, estimates for the relative improvement
        between models are computed, along with the p-value that
        the improvement is greater than zero.
        
    Parameters
    ----------
    scores : list or np.Array
        predicted probabilities of the positive class, either for
        a single model (1D array or flat list) or multiple (2D
        array with shape N_points x N_models)

    classes : list or np.Array
        binary observed values corresponding to each score

    metrics : list
        what performance metrics to use. options are AUC, RMSE,
        and recall

    recall_k : int
        what k to use in the recall computation, if needed.

    confidence_level : float
        what CIs to return

    n_bootstrap : int
        number of bootstrap iterations

    return_improvement : Bool
        whether to also return a relative improvement dataframe

    all_pairs : Bool
        whether to compute the difference for all pairs of
        models i.e. A vs B and B vs A.

    model_names : list or None
        human-readable model names to use as indices in the
        dataframe. If None, use Model 1, 2, etc.
        
    pval_method : `two-sided` or `one-sided`
        how to compute the bootstrap iprovement p-value, see
        the evaluate_improvement function for details
        
        
    Returns
    ----------
    performance_df : pd.DataFrame
        df that shows point estimates and confidence intervals
        for each of the models. rows are indexed by model, and columns
        are multi-indexed by metric then by "Estimate" and "CI"
        
    improvement_df : pd.DataFrame (optional)
        df that shows relative improvement between the models. rows
        are multi-indexed by first and second model, and columns are
        multi-indexed by metric then by "Estimate" and "p-value" (always
        using first model - second model)
    """

    #Convert inputs to arrays:
    if type(scores) == list: scores = np.array(scores)
    if type(classes) == list: classes = np.array(classes)
    N = len(classes)
    
    #Assign model names:
    if model_names is None:
        model_names = [f'Model {k+1}' for k,_ in enumerate(scores)] if len(scores.shape)==2 else ['Model']
    
    #Verify input:
    assert len(classes) == scores.shape[0]
    assert len(model_names) == scores.shape[1] if len(scores.shape)==2 else 1
    
    #Initialize the dictionary with our metrics:
    metrics_dict = performance_functions(recall_k)
    
    #Get the point estimates:
    estimates = {f:np.array([metrics_dict[f](classes, p) for p in scores.T]) for f in metrics}
    
    #Bootstrap the sample:
    boot_estimates = {f:[] for f in metrics}
    for _ in range(n_bootstrap):
    
        #Get the bootstrapped values:
        _classes, _scores = bootstrap_arrays(classes, scores)    
    
        #Compute the estimates:
        for f in metrics:
            _estimates_f = [metrics_dict[f](_classes, p) for p in _scores.T]
            boot_estimates[f].append(_estimates_f) 
    
    #Compute the CIs:
    q = [50*(1-confidence_level), 50*(1+confidence_level)]
    LBs = {f:np.nanpercentile(np.array(boot_estimates[f]), q[0], axis=0) for f in metrics}
    UBs = {f:np.nanpercentile(np.array(boot_estimates[f]), q[1], axis=0) for f in metrics}
    
    #Create the performance dataframe:
    performance_dict = {(f, col): d[f] for f in metrics for d,col in zip([estimates,LBs, UBs], ['Estimate', 'LB', 'UB'])}
    performance_df = pd.DataFrame(performance_dict, index=model_names)
                             
    #Create the improvement dataframe:
    if return_improvement:
        improvement_df = evaluate_improvement(boot_estimates,
                                              confidence_level,
                                              all_pairs,
                                              model_names,
                                              pval_method)
        
    return (performance_df, improvement_df) if return_improvement else performance_df

##########################################################################################################

def evaluate_improvement(estimates_dictionary,
                         confidence_level=0.95,
                         all_pairs=True,
                         model_names=None,
                         pval_method='one-sided'):
    """
    Evaluate the improvement between two models given
        the estimates for their performance on bootstrapped
        data.
        
    Parameters
    ----------
    estimates_dictionary : dict
        dictionary where keys are metrics and values are
        2D arrays of shape N_bootstrap x N_models

    confidence_level : float
        what CIs to return

    all_pairs : Bool
        whether to compute the difference for all pairs of
        models i.e. A vs B and B vs A.

    model_names : list or None
        human-readable model names to use as indices in the
        dataframe. If None, use Model 1, 2, etc.

    pval_method : `two-sided` or `one-sided`
        how to compute the relative improvement p-value.
        
        one-sided p_val = 1 - #(boot where model A > model B)/#boot
        two-sided p_val = 2*min(1-z, z) where z = quantile(0)
        
    Returns
    ----------
    improvement_df : pd.DataFrame
        df that shows relative improvement between the models. rows
        are multi-indexed by first and second model, and columns are
        multi-indexed by metric then by "Estimate" and "p-value" (always
        using first model 'improves upon' second model)
    """    
    #Collect quantiles:
    q = [50*(1-confidence_level), 50*(1+confidence_level)]
                             
    #We must do this one metric at a time:
    improvement_dict = {(f, col): [] for f in estimates_dictionary.keys() for col in ['Estimate', 'LB', 'UB', 'p-value']}
    for metric, arr in estimates_dictionary.items():
    
        #First let's get an array where the rows are the models:
        boot_estimates_arr = np.array(arr).T
    
        #Now select the first and second models:
        for idx_A, model_A in enumerate(boot_estimates_arr):
            for idx_B, model_B in enumerate(boot_estimates_arr):
    
                if (idx_A != idx_B if all_pairs else idx_A < idx_B):
    
                    #Get the differences:
                    deltas = model_A-model_B
                    #print(deltas)
        
                    #Get the estimates and CIs:
                    improvement_dict[(metric, 'Estimate')].append(np.nanmean(deltas))
                    improvement_dict[(metric, 'LB')].append(np.nanpercentile(deltas, q[0]))
                    improvement_dict[(metric, 'UB')].append(np.nanpercentile(deltas, q[1]))
        
                    #For one-sided p values, we should use the signal of improvement:
                    if pval_method == 'one-sided':
                        improvement = model_A < model_B if metric in ['MSE', 'RMSE'] else model_A > model_B
                        improvement_dict[(metric, 'p-value')].append(1 - np.nanmean(improvement.astype(int)))
                    elif pval_method == 'two-sided':
                        z = percentileofscore(deltas, 0, nan_policy='omit')/100
                        improvement_dict[(metric, 'p-value')].append(2*min(1-z, z))
    
    #Assign model names:
    if model_names is None:
        N_models = len(next(iter(estimates_dictionary.values()))[0])
        model_names = [f'Model {k+1}' for k in range(N_models)] if N_models > 1 else ['Model'] 
        
    #Assign model indices:
    model_index = []
    for idx_A, model_A in enumerate(model_names):
        for idx_B, model_B in enumerate(model_names):
            if (idx_A != idx_B if all_pairs else idx_A < idx_B):
                model_index.append((model_A,model_B))
    model_index=pd.MultiIndex.from_tuples(model_index, names=('Model A', 'Model B'))
    
    #Create a dataframe:
    improvement_df = pd.DataFrame(improvement_dict, 
                                  index=model_index)
    
    return improvement_df

##########################################################################################################

def performance_functions(recall_k=None):
    """
    Functions that compute a classification model's performance
        given the true classes (y) and the predictions (p), 
        respectively. Any new function implemented for performance
        must be added to this wrapper function, with additional
        arguments being fixed here. User will then select which
        metrics to compute in the function `evaluate_performance`.
    
    Parameters
    ----------
    recall_k : int or None
        when computing recall from probabilistic input p, top-k
        items will be turned into 1 and remaininig 0. If None,
        p must already be binary.
        
    Returns
    ----------
    metric_functions_dict : dictionary
        dictionary whose keys are available metrics (strings) and
        entries are two-argument functions that compute the metric
        for y,p.
    """

    #Compute AUC:
    AUC = lambda y,p: roc_auc_score(y,p)

    #Compute RMSE:
    RMSE = lambda y,p: mean_squared_error(y,p,squared=False)

    #Compute recall:
    def topk(p):
        if recall_k is None: return p
        else: return (p.argsort()[::-1] < recall_k).astype(int)
    recall = lambda y,p: recall_score(y, topk(p), zero_division=np.nan)

    return {'AUC': AUC, 'RMSE': RMSE, 'recall': recall}

##########################################################################################################

def bootstrap_arrays(classes, scores):
    """
    Bootstrap binary classes and 
        predictions for performance
        metrics.
    
    Parameters
    ----------
    classes, scores : np.Array
        arrays to bootstrap. The classes
        array will impose a constraint
        in the sample (the bootstrapped
        sample must contain two classes)
        
        
    Returns
    ----------
    boot_classes, boot_scores : np.Array
        bootstrapped arrays
    """

    #Get the length of the arrays:
    N = len(classes)
    assert len(scores)==N

    #Get a bootstraped sample with two classes:
    bootstrapped = False
    while not bootstrapped:
        boot_idx = resample(range(N), replace=True)
        boot_classes = classes[boot_idx]
        boot_scores = scores[boot_idx]
        if len(set(boot_classes)) != 1: bootstrapped=True

    return boot_classes, boot_scores

##########################################################################################################