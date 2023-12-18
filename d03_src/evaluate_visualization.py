import numpy as np
import pandas as pd
import geopandas as gpd
import arviz as az
from collections import defaultdict
from random import choices

import seaborn as sns
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib import colormaps
from matplotlib.lines import Line2D
plt.rcParams['mathtext.fontset'] = 'cm'

##################################################################################################################################

import sys
sys.path.append('../d03_src/')
from process_graph import generate_graph_census
from pygris.utils import erase_water
from vars import _covariates, _projected_crs, _park_shp
from utils import latex, annotate_params, weighted_median, cropcmap, set_figsize, add_psi

projected_crs = 'EPSG:2263'

##################################################################################################################################

def plot_chains_and_posteriors(all_inferred_params,
                               true_params=None,
                               passed_T=False, bins=100, density=False, burnin=0,
                               param_names=['theta0', 'theta1', 'psi'],
                               crop_plots=True,
                               ranges={'theta0': (-0.5, 0.5), 'theta1':(-0.1, 0.3)},
                               rhats=None):
    
    #Infer whether we passed a single chain or multiples:
    one_chain = True if (type(all_inferred_params)==dict or len(all_inferred_params)==1) else False
    if one_chain and len(all_inferred_params)!=1: all_inferred_params = [all_inferred_params]

    #Check if our true parameters include anything beyond T:
    if true_params is not None: only_true_T = False if param_names[0] in true_params.keys() else True
    
    #This avoids modifying the original dictionaries and excludes the array:
    all_inferred_params_for_df = [{param:chain[param] for param in chain if param != 'A'}
                                  for chain in all_inferred_params]
        
    #In case we passed an A array, we need to extract the mean!
    chains_with_A = 0
    for chain, chain_for_df in zip(all_inferred_params, all_inferred_params_for_df):
        
        if 'A' in chain:
            chains_with_A += 1
            chain_for_df['A_mean'] = [A_arr.mean()for A_arr in chain['A']]
            
            if true_params is not None:
                if not only_true_T: true_params['A_mean'] = np.mean(true_params['A'])
                if passed_T: true_params['T_mean'] = np.mean(true_params['T'])
                    
    if (chains_with_A == len(all_inferred_params)) and 'A_mean' not in param_names: param_names.append('A_mean')
    
    #Create a dataframe with our sampled data:
    chain_dfs = []
    for chain in all_inferred_params_for_df:
        chain_dfs.append(pd.DataFrame(chain))
        
    #Plot posteriors:
    fig = plt.figure(constrained_layout=True, figsize=(15,5*len(param_names)))
    Axs = fig.subplot_mosaic([[f'chain_{param}', f'hist_{param}'] for param in param_names],
                             gridspec_kw={'width_ratios':[1, 2]})
    
    for parameter in param_names:
        
        #Get the axes:
        ax_hist = Axs[f'hist_{parameter}']
        ax_chain = Axs[f'chain_{parameter}']
        
        p_range = (0, 1) if (parameter == 'psi' or parameter == 'A_mean') else (ranges[parameter] if parameter in ranges else None)
        
        #Plot each chain:
        for idx, chain_df in enumerate(chain_dfs):
            chain_df.iloc[burnin:].hist(column=parameter, ax=ax_hist, alpha=0.7, range=p_range, bins=41, label=f'chain {idx+1}')
            ax_chain.plot(chain_df[parameter], label=f'chain {idx+1}', alpha=0.7)
            
        #Mark true parameter values:
        if true_params is not None and not only_true_T:
            true_param_value = true_params[parameter]
            ax_hist.axvline(true_param_value, linestyle='--', color='black')
            ax_chain.axhline(true_param_value, linestyle='--', color='black')
        
        #We only add the legend in case we have multiple chains: 
        if not one_chain:
            _ = ax_hist.legend()
            _ = ax_chain.legend()

        #Labels go on the chain:
        ax_chain.set_xlabel('MCMC iteration')
        ax_chain.set_ylabel(latex(parameter))
        ax_chain.set_ylim(p_range)
        
        #Mark the burnin region
        if burnin > 0:
            ax_chain.axvline(burnin, linestyle='solid', color='red')
            ax_chain.axvspan(0, burnin, color='red', alpha=0.4)
        
        #Annotate the plot of A with the mean of T in case we passed:
        if parameter == 'A_mean' and passed_T and true_params is not None:
            print(f"Mean of T: {true_params['T_mean']}")
            ax_chain.axhline(true_params['T_mean'], linestyle=':',
                             color='green', alpha=0.9, label='T mean')
            ax_chain.annotate('mean of T', xy=(.9, .1), xycoords='axes fraction',
                              color='green', horizontalalignment='right')
            
        #Annotate the plot of A with the mean of T in case we passed:
        if rhats is not None and parameter in rhats:
            rhat = rhats[parameter]
            ax_chain.text(0.1, 0.1, r'$\hat{r} = $' + fr'${rhat:.3f}$',
                          ha="center", va="center", size=8,
                          bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

    return fig, Axs

def plot_pairwise(all_inferred_params, size=7.5, burnin=0):
    
    #Infer whether we passed a single chain or multiples:
    one_chain = True if (type(all_inferred_params)==dict or len(all_inferred_params)==1) else False
    if one_chain and len(all_inferred_params)!=1: all_inferred_params = [all_inferred_params]
    
    #Copy the dictionary and extract the mean:
    all_inferred_params_for_df = [{param:chain[param] for param in chain if param != 'A'}
                                  for chain in all_inferred_params]
    for chain, chain_for_df in zip(all_inferred_params, all_inferred_params_for_df):
        if 'A' in chain: chain_for_df['A_mean'] = [A_arr.mean()for A_arr in chain['A']]
    
    #Create a dataframe with our sampled data:
    inferred_params_df = pd.concat([pd.DataFrame(d)[burnin:]
                                    for d in all_inferred_params_for_df])
    n_params = len(inferred_params_df.columns)

    #Create a grid with n_params x n_params plots
    len_graph = n_params-1
    fig, Axes = plt.subplots(figsize=(len_graph*size, len_graph*size),
                             nrows=len_graph, ncols=len_graph)

    #Find the bounds for each parameter:
    all_bounds = inferred_params_df.agg([min, max])
    min_bounds = all_bounds.loc['min'].values
    max_bounds = all_bounds.loc['max'].values

    #Plot:
    for i in range(1, n_params):
        for j in range(len_graph):
            #We only plot lower triangular plots:
            if j < i:
                Axes[i-1][j].scatter(inferred_params_df.iloc[:,j], inferred_params_df.iloc[:,i], alpha=0.4)

                #If we have the first column, we add label to y_axis:
                if j == 0:
                    Axes[i-1][j].set_ylabel(latex(inferred_params_df.columns[i]), fontsize=2*size)

                #If we have the last row, we add label to x_axis
                if i == n_params-1:
                    Axes[i-1][j].set_xlabel(latex(inferred_params_df.columns[j]), fontsize=2*size)
                
                #If we have psi or A, enforce 0, 1 bounds:
                if inferred_params_df.columns[i] == 'psi' or inferred_params_df.columns[i] == 'A_mean':
                    Axes[i-1][j].set_ylim(0, 1)
                if inferred_params_df.columns[j] == 'psi' or inferred_params_df.columns[j] == 'A_mean':
                    Axes[i-1][j].set_xlim(0, 1)
                
                #Axes[i-1][j].set_xlim(min_bounds[j], max_bounds[j])
                #Axes[i-1][j].set_ylim(min_bounds[i], max_bounds[i])

            #If j >= i we have no plot:
            else:
                Axes[i-1][j].axis('off')
    
    plt.show()
    
    return Axes

def show_samples(sampled_A, sampled_T, size=(11,20)):
    
    fig, Axes = plt.subplots(figsize=size, ncols=2)
    
    Axes[0].imshow(np.array(sampled_A) == 1, aspect='auto', cmap='Greys')
    Axes[0].set_title("Inferred A")
    Axes[0].set_xticks([])
    
    Axes[1].imshow(np.array(sampled_T) == 1, aspect='auto', cmap='Greys')
    Axes[1].set_title("Inferred T")
    Axes[1].set_xticks([])
    
    plt.show()
    
    return Axes

def plot_Aproportions(sampled_A, real_A=None, real_T=None, size=(5,5), ax=None, show=True):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
    
    A_props = [A_arr.mean() for A_arr in sampled_A]
    ax.plot(A_props)
    
    if real_A is not None:
        real_A_prop = real_A.mean()
        ax.axhline(real_A_prop, linestyle='--', color='black')
    
    ax.axhline(0, linestyle='-', color='black')
    ax.axhline(1, linestyle='-', color='black') 
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(0)
    
    if show:
        plt.show()
        
    return ax

def plot_chain_stats_multiple(true_params_file,
                              sampled_params_file,
                              debug_params_file=None,
                              N=225,
                              n_plots=100,
                              n_sve_iter=1,
                              n_burnin=0, #use zero if the burn-in has already been removed
                              plot_pairwise_params=True,
                              plot_debugging_stats_general=False,
                              plot_debugging_stats_detailed=False,
                              crop_plots=True,
                              save_plots=False,
                              save_dir='plots/'):
    
    current_plot=0
    summaries = []
    print(f'Excluding {n_burnin} burn-in samples\n')
        
    if plot_debugging_stats_general:
        for (run, true_params_df), (run, sampled_params_df), (run, debug_params_df) in zip(true_params_file.groupby('run'),
                                                                                           sampled_params_file.groupby('run'),
                                                                                           debug_params_file.groupby('run')):

            print(run)

            true_psi = true_params_df.iloc[0].psi
            true_theta0 = true_params_df.iloc[0].theta0
            true_theta1 = true_params_df.iloc[0].theta1
            true_params_dict = {'psi': true_psi,
                                'theta0': true_theta0,
                                'theta1': true_theta1,
                                'A':true_params_df.loc[:,[f'A{k+1}' for k in range(N)]].values.flatten()/2 + 0.5,
                                'T':true_params_df.loc[:,[f'T{k+1}' for k in range(N)]].values.flatten()}

            for (parameter_run, sampled_params_in_this_run_df) in sampled_params_df.groupby('parameter_run'):
                
                sampled_params_dict = sampled_params_in_this_run_df[['psi', 'theta0', 'theta1']].to_dict(orient='list')
                sampled_params_dict['A'] = sampled_params_in_this_run_df.loc[:,[f'A{k+1}' for k in range(N)]].values/2 + 0.5 
                
                print("inferred posteriors")
                summary_df = summarize_params(sampled_params_dict, true_params_dict, show=True, burnin=n_burnin)
                fig, Ax = plot_chains_and_posteriors(sampled_params_dict, true_params_dict, passed_T=True, burnin=n_burnin)
                if save_plots: fig.savefig(f'{save_dir}chain{current_plot}.pdf',format='pdf')
                if plot_pairwise_params:
                    _ = plot_pairwise(sampled_params_dict, burnin=n_burnin)
                summaries.append(summary_df)
                break

            debug_params_filtered = debug_params_df.drop(debug_params_df.index[::n_sve_iter+1]).iloc[n_burnin:] #do this to remove the zero-th iteration
            print('Histograms of the ratio in the SVEA:')
            a = debug_params_filtered.apply(lambda row: min(row['a'], 1), axis=1)

            fig, ax = plt.subplots(figsize=(15, 5))
            ax.hist(a, label='acceptance probability', alpha=1)
            ax.set_title('Histogram of acceptance probabilities')
            plt.show()
            print(a.describe())
            print('\n')

            if plot_debugging_stats_detailed:

                log_densities = debug_params_filtered['density']
                log_priors = debug_params_filtered['prior']
                log_densities_w = debug_params_filtered['density_w']

                log_densities_proposal = debug_params_filtered['density_p']
                log_priors_proposal = debug_params_filtered['prior_p']
                log_densities_w_proposal = debug_params_filtered['density_w_p']

                fig, ax = plt.subplots(figsize=(15, 5))
                for arr, lbl in zip([log_densities_proposal, log_priors_proposal, log_densities_w],
                                    ['density (proposed)', 'prior (proposed)', 'auxiliary density']):
                    ax.hist(arr, label=lbl, alpha=0.3, bins=50)
                ax.set_title('Log-densities of terms in the numerator')
                ax.legend()
                plt.show()

                fig, ax = plt.subplots(figsize=(15, 5))
                for arr, lbl in zip([log_densities, log_priors, log_densities_w_proposal],
                                    ['density', 'prior', 'auxiliary density (proposed)']):
                    ax.hist(arr, label=lbl, alpha=0.3, bins=50)
                ax.set_title('Log-densities of terms in the denominator')
                ax.legend()
                plt.show()

                fig, ax = plt.subplots(figsize=(15, 5))
                ax.hist(np.log(debug_params_filtered.a[debug_params_filtered.a != 0]), bins=50)
                ax.set_title('Log-acceptance probability (not truncated)')
                plt.show()

                print('\n')
                print('Auxiliary variables that were used:')
                w_means = np.mean(debug_params_filtered.loc[:,[f'A{k+1}' for k in range(N)]].values, axis=1)/2 +0.5
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.plot(w_means)
                ax.set_title('mean of the auxiliary variable used in the SVEA per iteration')
                plt.show()

            print('\n\n')
            current_plot +=1
            
            if current_plot >= n_plots:
                break
                
    else:
        for (run, true_params_df), (run, sampled_params_df) in zip(true_params_file.groupby('run'),
                                                                   sampled_params_file.groupby('run')):

            print(run)

            true_psi = true_params_df.iloc[0].psi
            true_theta0 = true_params_df.iloc[0].theta0
            true_theta1 = true_params_df.iloc[0].theta1
            true_params_dict = {'psi': true_psi,
                                'theta0': true_theta0,
                                'theta1': true_theta1,
                                'A':true_params_df.loc[:,[f'A{k+1}' for k in range(N)]].values.flatten()/2 + 0.5,
                                'T':true_params_df.loc[:,[f'T{k+1}' for k in range(N)]].values.flatten()}

            for (parameter_run, sampled_params_in_this_run_df) in sampled_params_df.groupby('parameter_run'):
                
                sampled_params_dict = sampled_params_in_this_run_df[['psi', 'theta0', 'theta1']].to_dict(orient='list')
                sampled_params_dict['A'] = sampled_params_in_this_run_df.loc[:,[f'A{k+1}' for k in range(N)]].values/2 + 0.5
                
                print("inferred posteriors")
                summary_df = summarize_params(sampled_params_dict, true_params_dict, show=True, burnin=n_burnin)
                fig, Ax = plot_chains_and_posteriors(sampled_params_dict, true_params_dict, passed_T=True, burnin=n_burnin)
                if save_plots: fig.savefig(f'{save_dir}chain{current_plot}.pdf',format='pdf')
                if plot_pairwise_params:
                    _ = plot_pairwise(sampled_params_dict, burnin=n_burnin)
                summaries.append(summary_df)
                break
            
            print('\n\n')
            current_plot +=1
            if current_plot >= n_plots:
                break
            
    return summaries


                            
########################################################################################

def plot_calibration(summaries,
                     param_names=['theta0', 'theta1', 'psi'],
                     ax=None,
                     figsize=set_figsize(height_ratio=1),
                     show=True,
                     remove_axes=True,
                     title='Calibration of the model per parameter',
                     title_size=16,
                     xlabel='Confidence Interval (%)',
                     xlabel_size=14,
                     ylabel='Coverage (%)',
                     ylabel_size=14,
                     leg_size=14,
                     leg_ncols=3,
                     ticks_size=12):
    
    cis = [k/20 for k in range(21)]
    markers = {param:[] for param in param_names}
    
    for qt in cis:
        def get_pct_str(val):
            pct_str = f'{val:.0%}' if round(100*val,1)%1 <= 0.01 else f'{val:.1%}'
            return pct_str

        lower_bound = get_pct_str((1-qt)/2)
        upper_bound = get_pct_str((1+qt)/2)
        is_within_interval = {param:0 for param in param_names}
        
        for summary_df in summaries:
            for param in param_names:
                true_param = summary_df.loc[param, 'true']
                if true_param <= summary_df.loc[param, upper_bound] and true_param >=summary_df.loc[param, lower_bound]:
                    is_within_interval[param] += 1
        
        pct_within_interval = {param: 100*is_within_interval[param]/len(summaries) for param in param_names}
        
        for param in param_names:
            markers[param].append((100*qt, pct_within_interval[param]))
    
    #Make the plot:
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    ax.axline((0, 0), slope=1, color='k', linestyle='--', alpha=0.6)
    for param in param_names:
        true, computed = zip(*markers[param])
        ax.plot(true, computed, marker='x', label=latex(param), markersize=3)
        
    ax.set_title(title, fontsize=title_size)
    ax.set_xlabel(xlabel, fontsize=xlabel_size)
    ax.set_ylabel(ylabel, fontsize=ylabel_size)
    ax.legend(ncol=leg_ncols, loc='lower right', fontsize=leg_size)

    ax.tick_params(labelsize=ticks_size)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')              
    #if remove_axes: _ = ax.spines[['right', 'top']].set_visible(False)
    if show: plt.show()

    return ax

def plot_evaluation_scatterplot(evaluation_summary,
                                range=None,
                                title=None,
                                title_size=15,
                                xlabel='True parameter',
                                xlabel_size=12,
                                ylabel='Posterior mean',
                                ylabel_size=12,
                                correlation=True,
                                correlation_size=12,
                                alpha=0.5,
                                ticks_size=10,
                                color='black',
                                ax=None,
                                figsize=(7,7),
                                show=True):

    #Create the figure:
    if ax is None: fig, ax = plt.subplots(figsize=figsize)

    #Get the arrays:
    true = evaluation_summary['True'].values
    estimates = evaluation_summary['Estimate'].values
    UB = evaluation_summary['Upper bound'].values
    LB = evaluation_summary['Lower bound'].values
                                    
    #Get the error bars:                             
    CIs = np.vstack([estimates - LB, UB - estimates])                  

    #Plot:
    _ = ax.errorbar(x=true, y=estimates, yerr=CIs,
                    ecolor='grey', capsize=2, elinewidth=1, linewidth=0,
                    markersize=3, marker='D', mfc=color, mec=color, alpha=alpha)

    #Add the diagonal line:
    _ = ax.axline((0, 0), slope=1, color='k', linestyle='--', zorder=-1)
    
    #Add the correaltions:
    if correlation:
        corr = np.corrcoef(true, estimates)[0, 1]
        _ = ax.text(0.8, 0.1, r'$\rho$'+ f' = {corr:.2f}',
                    ha="center", va="center", size=correlation_size, transform=ax.transAxes,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))
        
    #Configure the axes:
    _ = ax.set_title(title, fontsize=title_size)
    _ = ax.set_xlabel(xlabel, fontsize=xlabel_size)
    _ = ax.set_ylabel(ylabel, fontsize=ylabel_size)
    _ = ax.tick_params(labelsize=ticks_size)

    #Configure the range:
    if range is None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        range = (min([xlim[0], ylim[0]]), max([xlim[1], ylim[1]]))
    ax.set_xlim(range[0], range[1])
    ax.set_ylim(range[0], range[1])

    if show: plt.show()
        
    return ax

########################################################################################

def plot_AUC(fpr_list,
             tpr_list,
             AUC_list=None,
             average_AUCs=False, labels=None,
             figsize=set_figsize(height_ratio=1),
             ax=None):

    #Ensure we have lists:
    if type(fpr_list[0]) == float: fpr_list = [fpr_list]
    if type(tpr_list[0]) == float: tpr_list = [tpr_list]
    if type(AUC_list)    == float: AUC_list = [AUC_list]
    if labels is None:             labels = [None]*len(fpr_list)
    assert len(fpr_list)==len(tpr_list)==len(AUC_list)==len(labels)

    #Create the ax and add the baseline:
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    ax.axline((0, 0), slope=1, color='k', linestyle='--', label='random prediction')

    #Plot each ROC (adjust transparency and color scheme 
    # depending on plot vs not plot)
    for fpr, tpr, label in zip(fpr_list, tpr_list, labels):
        _ = ax.plot(fpr, tpr,
                    color='red' if labels[0] is None else None,
                    alpha=1. if not average_AUCs else 0.2,
                    label=label if (len(labels)!=1 or labels[0]!=None) else 'model prediction')

    #Add an AUC score if passed:
    if AUC_list is not None:

        #Annotate with the average AUC from all curves:
        if average_AUCs:
            annotation = f'avg. AUC = {np.nanmean(AUC_list):.3f}'
            #Plot an average ROC:
            _ = ax.plot(np.array(fpr_list).mean(axis=0), np.array(tpr_list).mean(axis=0),
                        color='red', alpha=1, linewidth=2, label='average ROC')
        
        #Annotate with labels:
        elif labels[0] is not None:
            annotation = '\n'.join([f'AUC ({label}) = {AUC:.3f}'
                                    for label, AUC in zip(labels, AUC_list)])
        
        #Annotate with the single AUC:
        elif len(AUC_list) == 1:
            annotation = f'AUC = {AUC_list[0]:.3f}'
            
        #Skip annotation:
        else:
            annotation = None
            
        #Annotate:
        _ = ax.text(0.2, 0.9, annotation,
                    ha="center", va="center", size=10, transform=ax.transAxes,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))
        
    #Fix ax parameters:
    _ = ax.set_xlabel('False Positive Rate')
    _ = ax.set_ylabel('True Positive Rate')
    _ = ax.legend(loc='lower right')
    _ = ax.set_title('ROC Curves for ground-truth predictions')
    ax.set_aspect('equal')

    return ax

def plot_classifier_calibration(true_prob_list,
                                pred_prob_list,
                                labels=None,
                                ax=None,
                                figsize=set_figsize(height_ratio=1)):

    #Ensure we have lists:
    if type(true_prob_list[0]) == float: true_prob_list = [true_prob_list]
    if type(pred_prob_list[0]) == float: pred_prob_list = [pred_prob_list]
    if labels is None: labels = [None]*len(pred_prob_list)
    assert len(true_prob_list)==len(pred_prob_list)==len(labels)

    #Create the ax and add the baseline:
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    ax.axline((0, 0), slope=1, color='k', linestyle='--', label='Perfectly calibrated')

    #Plot each curve (adjust transparency and color scheme)
    for true_prob_arr, pred_prob_arr, label in zip(true_prob_list, pred_prob_list, labels):
        _ = ax.plot(pred_prob_arr, true_prob_arr,
                    marker='s',
                    color='red' if labels[0] is None else None,
                    alpha=0.2   if labels[0] is None else 1.,
                    label=label if (len(labels)!=1 or labels[0]!=None) else 'Model')

    #Fix ax parameters:
    _ = ax.set_xlim(-0.05, 1.05)
    _ = ax.set_ylim(-0.05, 1.05)
    _ = ax.set_xlabel('Mean predicted probability for positive nodes')
    _ = ax.set_ylabel('Fraction of positive nodes')
    _ = ax.legend(loc='lower right')
    _ = ax.set_title('Classifier Calibration Curves')

    return ax

def plot_rhatdistributions(rhats):
    fig, ax = plt.subplots(figsize=(15, 5))
    for idx, (param, vals) in enumerate(rhats.items()):
        ax.hist(vals, range=(1, 2), bins=25, label=latex(param), alpha=0.5)
    _ = ax.legend()
    _ = ax.set_title(r'Distribution of $\hat{r}$ per parameter over ' + f'{len(vals)} experiments')
    _ = ax.set_xlabel(r'$\hat{r}$ value', fontsize=13)
    _ = ax.set_ylabel(r'Frequency')

    return ax

def plot_Aprob(A_prob,
               figsize=set_figsize(height_ratio=1), ax=None,
               unit='tracts', remove_long_edges=True, tresh_edgelength=2000,
               projected_crs=projected_crs):
                   
    #Collect the graph and GeoDataFrames:               
    census_gdf, graph, census_gdf_raw = generate_graph_census(unit, remove_long_edges=remove_long_edges, tresh_edgelength=tresh_edgelength) 
                   
    #Plot the base GeoDataFrame:
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    ax = erase_water(census_gdf_raw).to_crs(projected_crs).plot(ax=ax, color='darkgray')

    #Plot the probabilities:
    census_gdf['inf_Aprob'] = A_prob
    ax = erase_water(census_gdf).to_crs(projected_crs).plot(ax=ax, column='inf_Aprob', vmin=0, vmax=1, cmap='viridis')

    _ = ax.axis('off')
    _ = ax.set_title('Inferred Probability', fontsize=25)
                   
    return ax

def plot_95CI(summary, annotate=True, covariates=_covariates, crop_plots=True):
    
    #Select the parameters:
    param_names = [param for param in summary.index if param != 'A_mean']

    #Get medians, upper, and lower bounds:
    medians = summary.loc[param_names[::-1], '50%'].values
    UB = summary.loc[param_names[::-1], '97.5%'].values
    LB = summary.loc[param_names[::-1], '2.5%'].values

    #Build CIs:
    CIs = np.vstack([medians-LB, UB-medians])
    fig, ax = plt.subplots(figsize=(7, 5))

    #Plot:
    _ = ax.errorbar(x=medians, y=param_names[::-1], xerr=CIs,
                    color='black', capsize=3, linestyle='None', linewidth=1,
                    marker="o", markersize=5, mfc="black", mec="black")
    #Grids:
    _ = ax.axvline(0, linestyle='--', color='k', alpha=0.5, zorder=-1)
    _ = ax.axvline(0.5, linestyle='--', color='k', alpha=0.2, zorder=-1)
    _ = ax.axvline(-0.5, linestyle='--', color='k', alpha=0.2, zorder=-1)

    #Set ticks and labels:    
    _ = ax.set_ylim(-1, len(param_names))
    _ = ax.set_yticks(range(len(param_names)),[latex(param) for param in param_names[::-1]], fontsize=15)
    if crop_plots: _ = ax.set_xlim(-1, 1)

    #Set title:
    _ = ax.set_title('95% CIs for estimated parameters\n')

    #Anotate:
    if annotate:
        for idx, param in enumerate(param_names[::-1]):
            _ = ax.annotate(annotate_params(param, covariates), xy=(.98, idx-0.4), xycoords='data',
                            color='red', alpha=0.7, horizontalalignment='right')

    return ax

def plot_CIs_covariates(CIs_df, crop_plots=True, ax=None, figsize=(6, 6),
                        covariate_names=None, show=True, ylabel_size=12,
                        xlabel_size=12,
                        color_CIs_by_significance=True, fill_between=False, horizontal_lines=True):
    
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    
    CIs_df = CIs_df.replace({'-':np.nan}).dropna()
    if covariate_names is None: covariate_names = CIs_df.index.values
    
    estimates = CIs_df.loc[covariate_names, 'Estimate'].values
    UB = CIs_df.loc[covariate_names, 'Upper bound'].values
    LB = CIs_df.loc[covariate_names, 'Lower bound'].values
    CIs = np.vstack([estimates-LB, UB-estimates])

    #Collect colors:
    if color_CIs_by_significance:
        colors = ['red' if ub < 0 else ('blue' if lb > 0 else 'grey')
                 for ub, lb in list(zip(UB, LB))]
    else:
        colors = ['black']*len(estimates)

    #Plot:
    for estimate, name, CI, color in zip(estimates[::-1],
                                         covariate_names[::-1],
                                         CIs[::, ::-1].T,
                                         colors[::-1]):
        _ = ax.errorbar(x=estimate,
                        y=name,
                        xerr=CI.reshape(2,1),
                        ecolor=color,
                        capsize=5,
                        linestyle='None',
                        linewidth=1.5,
                        marker="D",
                        markersize=5,
                        mfc=color,
                        mec=color)

    ax.tick_params(axis='y', labelsize=ylabel_size)
    ax.tick_params(axis='x', labelsize=xlabel_size)
    #Grids:
    xlim = ax.get_xlim()
    _ = ax.axvline(0, linestyle='--', color='black', alpha=0.75, zorder=-1, linewidth=1.)

    #for val in [0.5, 1, 1.5, 2]:
    #    if xlim[1] > val: _ = ax.axvline(val, linestyle='-', color='lightgrey', alpha=0.5, zorder=-1, linewidth=1)
    #for val in [-0.5, -1, -1.5, -2]:
    #    if xlim[0] < val: _ = ax.axvline(val, linestyle='-', color='lightgrey', alpha=0.5, zorder=-1, linewidth=1)

    #Fill between:
    if fill_between:
        #Home ownership:
        _ = ax.fill_between(np.linspace(xlim[0], xlim[1], 1000, endpoint=True), -0.5, 1.5, alpha=.4, color='lightgrey', linewidth=0)
        #Education:
        _ = ax.fill_between(np.linspace(xlim[0], xlim[1], 1000, endpoint=True), 6.5, 8.5, alpha=.4, color='lightgrey', linewidth=0)
        #Demographics:
        _ = ax.fill_between(np.linspace(xlim[0], xlim[1], 1000, endpoint=True), 10.5, 14.5, alpha=.4, color='lightgrey', linewidth=0)
    if horizontal_lines:
        for y, c in enumerate(colors[::-1]):
            _ = ax.hlines(y=y, xmin=-3, xmax=3, linestyle='--', linewidth=0.5, alpha=0.5, color='grey')
    _ = ax.set_xlim(xlim)
    _ = ax.set_ylim(-0.5, len(estimates)-0.5)
            
    if show: plt.show()
        
    return ax
                            
############################################

def correlation_pairlot(value_list,
                        value_names=None,
                        plot_correlations=True,
                        correlation_fontsize=12,
                        label_fontsize=30,
                        alpha=0.5):

    #From values and their names, do a DataFrame:
    plot_df = pd.DataFrame(data=value_list, index=value_names).T

    #Plot:
    sns.set_context('paper', rc={'axes.labelsize':label_fontsize})
    graph = sns.pairplot(plot_df, corner=True, plot_kws={'alpha':alpha})

    #Apply correlations:
    if plot_correlations:
        
        def corrfunc(x, y, ax=None, **kwargs):
            r, _ = spearmanr(x, y, nan_policy='omit')
            ax = ax or plt.gca()
            ax.annotate(fr'$\rho$ = {r:.2f}', xy=(.6, .1),
                        size=correlation_fontsize,
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=.5'),
                        xycoords=ax.transAxes)
            
        _ = graph.map_lower(corrfunc)
                            
    return graph
                            
def plot_psi_correlation(estimates_x,
                         estimates_y,
                         covariates,
                         intercept=True,
                         correlation=True,
                         correlation_fontsize=15,
                         ticks_x=True,
                         ticks_y=True,
                         ticks_fontsize=15,
                         label_x=None,
                         label_y=None,
                         label_fontsize=25,
                         title=None,
                         title_fontsize=30,
                         alpha=0.25,
                         ax=None,
                         size=10,
                         show=False):
    """
    Plots the correlation between inferred report rates
        according to two distinct models. To set the unit
        of analysis equal across both axis, one of the
        models is projected onto the other---determined
        by which covariate arrays are provided.
    """

    #Get the psi_values:
    psi_x = add_psi(estimates_x, covariates, intercept)
    psi_y = add_psi(estimates_y, covariates, intercept)

    #Plot:
    if ax is None: fig, ax = plt.subplots(figsize=(size,size))
    _ = ax.scatter(psi_x, psi_y, alpha=alpha)

    #Find the correlation:
    if correlation:
        r, _ = spearmanr(psi_x, psi_y, nan_policy='omit')
        _ = ax.annotate(fr'$\rho$ = {r:.2f}', xy=(.65, .1),
                        size=correlation_fontsize,
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=.5'),
                        xycoords=ax.transAxes)
        
    #Adjust the ticks:
    ticks=[0., .2, .4, .6, .8, 1.]
    _ = ax.set_xticks(ticks, ticks) if ticks_x else ax.set_xticks([], [])
    _ = ax.set_yticks(ticks, ticks) if ticks_y else ax.set_yticks([], [])
    _ = ax.tick_params(axis='both', which='major', labelsize=ticks_fontsize)
    _ = ax.set_xlim(-0.02, 1.02)
    _ = ax.set_ylim(-0.02, 1.02)

    #Adjust the labels:
    _ = ax.set_xlabel(label_x, size=label_fontsize)
    _ = ax.set_ylabel(label_y, size=label_fontsize)
                             
    #Adjust the axis:
    _ = ax.set_title(title, fontsize=title_fontsize)
    _ = ax.set_aspect('equal')
    _ = ax.spines[['right', 'top']].set_visible(False)
                             
    if show: plt.show()
                             
    return ax
                                        
############################################
def plot_Aprob_bar(col_to_plot,
                   n_class=None, class_upper=[0.4, 0.6, 1.], class_labels=None,
                   bar_width=0.9,
                   max_val=1,
                   ax=None, figsize=(5,3), show=False,
                   cmap=None, cmap_name='YlGnBu', color_center=True,
                   reports=True,
                   report_color=None,
                   report_centroid=False, report_hatch=False, report_marker_color='white',
                   title=None, xlabel=None):

    #Get the upper class limits if not provided:
    if class_upper is None: class_upper = [(k+1)/n_class for k in range(n_class)]
    n_class = len(class_upper)

    #Get class names:
    if class_labels is None:
        if n_class == 2:
            class_labels = ['Low\nProbability', 'High\nProbability']
        elif n_class == 3:
            class_labels = ['Low\nProbability', 'Medium\nProbability', 'High\nProbability']
        elif n_class == 4:
            class_labels = ['Low\nProbability', 'Medium Low\nProbability', 'Medium High\nProbability', 'High\nProbability']
        else:
            class_labels = [f'< {UL:.2f} Probability' for UL in class_upper]
    
    #Classify:
    classes = pd.cut(col_to_plot, bins=[0.]+class_upper, include_lowest=True, right=False, labels=class_labels)
    classes_dict = classes.value_counts().to_dict()
    if reports: classes_dict['Reported'] = sum(classes.isna())

    #Create the ax:
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
                       
    #Get the colors:     
    norm = plt.Normalize(0., max_val)              
    if cmap is None: cmap=colormaps.get_cmap(cmap_name) #prob_color
    class_center = (np.array(class_upper) + np.array([0.] + class_upper[:-1]))*0.5
    colors = [cmap(norm(k)) for k in (class_center if color_center else class_upper)]
    if report_color is None: report_color=cmap(1.)
    colors.append(report_color)

    #Plot:
    y_label, width = list(classes_dict.keys()), list(classes_dict.values())
    _ = ax.barh(y=range(len(width)), width=width, color=colors, height=bar_width, zorder=1)
    _ = ax.set_yticks(range(len(width)), y_label)

    #Annotate the reports bar:
    if reports:
        if report_centroid:
            _ = ax.scatter(x=classes_dict['Reported']/2, y=n_class,
                           color=report_marker_color, marker='o')
        elif report_hatch:
            _ = ax.barh(n_class, width=classes_dict['Reported'],
                        color=report_color, hatch='////', edgecolor=report_marker_color,
                        linewidth=0.5, height=bar_width, zorder=2)

    #Remove the axis:
    _ = ax.spines[['right', 'top']].set_visible(False)

    #Set ticks:
    _ = ax.set_xticks(width)
    _ = ax.vlines(width, ymin=-1, ymax=[k+bar_width/2 for k in range(n_class+1)],
                  zorder=0, linestyle='--', color='grey', linewidth=0.75)
    _ = ax.set_ylim(-0.6, n_class+0.6)             
    _ = ax.set_title(title)
    _ = ax.set_xlabel(xlabel, loc='right')
    if show: plt.show()
    
    return ax
                       
def plot_hist(values,
              weights=None,
              n_bins=25,
              range=(0,1),
              title=None,
              ylabel=None,
              ylabel_size=18,
              xlabel=None,
              xlabel_size=18,
              ticks_size=14,
              fixed_color=None, cmap=None, cmap_name='copper_r',
              orientation='vertical',
              special_class=False,
              special_value=1.,
              special_color='maroon',
              special_centroid=False,
              special_hatch=False,
              special_marker_color='white',
              special_hline=False,
              figsize=(6,3.5), ax=None, show=False):
    
    if ax is None: fig, ax = plt.subplots(figsize=figsize)

    #Plot the histogram:
    counts, bins, patches = ax.hist(values, bins=n_bins, range=range,
                                    edgecolor='white', linewidth=2,
                                    color=fixed_color, orientation=orientation,
                                    weights=weights, zorder=1)

    #Color the bins:
    if not fixed_color:
        if cmap is None: cmap=colormaps.get_cmap(cmap_name)
        norm = plt.Normalize(range[0], range[1])
        for count, bin_value, rectangle in zip(counts, bins[1:], patches):
            if (bin_value != special_value) or not special_class:
                _ = plt.setp(rectangle, facecolor=cmap(norm(bin_value)))
            else:
                new_rectangle = mpatches.Rectangle(xy=rectangle.get_xy(),
                                                   width=rectangle.get_width(),
                                                   height=rectangle.get_height(),
                                                   facecolor=special_color,
                                                   linewidth=0.5,
                                                   hatch='////' if special_hatch else None,
                                                   edgecolor=special_marker_color,
                                                   zorder=2)
                #Add the centroid:
                if special_centroid: _ = ax.scatter(new_rectangle.center, color=special_marker_color, marker='o')
                #Add a line for the special bar:
                if special_hline: _ = ax.hlines(y=count, xmin=0, xmax=special_value, zorder=0, linestyle='--', color='grey', linewidth=0.75)
                _ = ax.add_patch(new_rectangle)
            
    #Customize axis:
    if orientation == 'vertical': _ = ax.set_xlim(range)
    if orientation == 'horizontal': _ = ax.set_ylim(range)
    _ = ax.spines[['right', 'top']].set_visible(False)
    _ = ax.set_title(title)
    _ = ax.set_ylabel(ylabel, ha='left', y=1.05, rotation=0, labelpad=0, fontsize=ylabel_size)
    _ = ax.set_xlabel(xlabel, loc='center', fontsize=xlabel_size)
    _ = ax.ticklabel_format(useOffset=False, style='plain')
    _ = ax.tick_params(labelsize=ticks_size)
    if show: plt.show()

    return ax

def plot_psi_bar(values,
                 discrete_categories=None, weights=None, class_labels=None,
                 aggregation='mean',
                 bar_width=0.7, ax=None, figsize=(6,4), show=False,
                 fixed_color=None, color_by_class=False, cmap=None, cmap_name='copper_r',
                 relative_to_city=False,
                 draw_hlines=False,
                 title=None,
                 ylabel=None,
                 ylabel_size=18,
                 xlabel_size=18,
                 ticks_size=14):

    #We must pass either discrete categories xor weights:
    assert (discrete_categories is not None)^(weights is not None)

    #If discrete_categories are passed:
    if discrete_categories is not None:
        #Get the labels and verify input:
        if class_labels is None: class_labels = set(discrete_categories)
        assert len(discrete_categories) == len(values)
        #Pandas to groupby:
        grouped_df = pd.DataFrame([values, discrete_categories], ['values', 'categories']).T.groupby('categories')
        #Compute mean or median:
        if aggregation == 'mean': bar_dict = grouped_df['values'].mean()[class_labels].to_dict()
        if aggregation == 'median': bar_dict = grouped_df['values'].median()[class_labels].to_dict()
        #Normalize for the city:
        if relative_to_city:
            if aggregation == 'mean': city_baseline = values.mean()
            if aggregation == 'median': city_baseline = values.median()
            bar_dict = {group: 100*val/city_baseline - 100 for group, val in bar_dict.items()}
        
    #If weights are passed:
    elif weights is not None:
        #Get the labels and verify input:
        if class_labels is None: class_labels = range(len(weights.T))
        assert weights.shape == (len(values), len(class_labels))
        #Compute the weighted mean or median in every group:
        if aggregation == 'mean': bar_dict = {group: np.average(values, weights=weights[:,k]) for k, group in enumerate(class_labels)}
        if aggregation == 'median': bar_dict = {group: weighted_median(values, weights=weights[:,k]) for k, group in enumerate(class_labels)}
        #Normalize for the city:
        if relative_to_city:
            if aggregation == 'mean': city_baseline = np.average(values, weights=weights.sum(axis=1))
            if aggregation == 'median': city_baseline = weighted_median(values, weights=weights.sum(axis=1))
            bar_dict = {group: 100*val/city_baseline - 100 for group, val in bar_dict.items()}

    #Create the ax:
    if ax is None: fig, ax = plt.subplots(figsize=figsize)

    #Get the colors:
    if cmap is None: cmap=colormaps.get_cmap(cmap_name)
    #If we passed a fixed color for all the bars:
    if fixed_color: colors = [fixed_color for group in class_labels]
    #In case we want to color each bar according to the class:
    elif color_by_class: colors = [cmap(k/len(class_labels)) for k,_ in enumerate(class_labels)]
    #In case we normalize by the city, we move the middle point to zero:
    elif relative_to_city: colors = [(cmap(1.) if bar_dict[group]>=0. else cmap(0.)) for group in class_labels]
    #In case we want to color each bar according to its height:
    else: colors = [cmap(bar_dict[group]) for group in class_labels]

    #Plot:
    height = list(bar_dict.values())
    _ = ax.bar(x=range(len(height)), height=height, color=colors, zorder=1, width=bar_width, edgecolor='white', linewidth=0.75)
    _ = ax.set_xticks(range(len(height)), [group.capitalize() for group in class_labels], rotation=45, fontsize=xlabel_size)

    #Remove the axis:
    _ = ax.spines[['right', 'top']].set_visible(False)
    if relative_to_city: _ = ax.spines['bottom'].set_position(('data', 0))

    #Set ticks:
    #_ = ax.set_yticks()#height, np.round(height,2))
    if draw_hlines:
        _ = ax.hlines(height, xmin=-1, xmax=[k+bar_width/2 for k in range(len(class_labels))],
                      zorder=0, linestyle='--', color='grey', linewidth=0.75)
    _ = ax.set_xlim(-0.6, len(class_labels)+0.6)             
    _ = ax.set_title(title, size=10, loc='left')
    _ = ax.set_ylabel(ylabel, ha='left', y=1.05, rotation=0, labelpad=0, fontsize=ylabel_size)
    _ = ax.tick_params(labelsize=ticks_size)
    if show: plt.show()

    return ax
                     
##############################
def map(values,
        values_vmin=0,
        values_vmax=1,
        values_name='Inferred Probability',
        gdf=None,
        base_gdf=None,
        unit='census',
        exclude_parks=True,
        park_shp=_park_shp,
        projected_crs=_projected_crs,
        cmap=None,
        cmap_name='YlGnBu',
        cmap_cropvalue=None,
        cmap_cropvalue_min=None,
        special_class=True,
        special_value=1.,
        special_color=None,
        special_centroid=False,
        special_hatch=False,
        special_line=False,
        special_marker_color='white',
        special_label='Report observed',
        bar_plot=False,
        bar_n_class=None,
        bar_class_upper=[0.4, 0.6, 1.],
        bar_class_labels=None,
        hist_plot=False,
        hist_bins=25,
        hist_fixedcolor=None,
        hist_orientation='vertical',
        hist_weights=None,
        hist_ylabelsize=18,
        hist_xlabelsize=18,
        psibar_classes=['white', 'black', 'hispanic', 'asian'],
        psibar_categories=None,
        psibar_weights=None,
        psibar_hlines=True,
        psibar_ylabelsize=18,
        psibar_xlabelsize=18,
        ticks_size=14,
        inset_dims=[0.08, 0.75, 0.38, 0.18],
        inset_colorbar=False,
        legend_nodata=True,
        title='Probability of a Street Flood during Hurricane Ida (September 1st, 2021)',
        show=True,
        save=False,
        save_dir='../d07_plots/',
        save_name='',
        ax=None):
    """
    Wrapper function for a map (mainly configured for NYC) of a continuous variable and,
      potentially, categorical annotations or a subplot (bar counts or histogram)

    Parameters
    ----------
    col_to_plot : iterable
        list of values for the variable to map, must be of the
        same dimension as the provided gdf
        
    gdf : geopandas.GeoDataFrame or None
        geodataframe with regions of interest in the map. If None,
        will collect the NYC census tract gdf 
        
    base_gdf : geopandas.GeoDataFrame or None
        full geodataframe if different than mappable gdf
    projected_crs : geopandas.crs
        coordinate reference system, in projected units, to perform
        spatial operations and map
        
    cmap : matplotlib.colormap or None
        colormap to use. If None, get from name 
    cmap_name : str, default 'YlGnBu
        colormap to use in str name

    special_class : bool, default True
        whether some polygons must be highlighted from the
        other. Used for reported tracts in the probability plot.
    special_value : float or iterable, default 1.
        value of polygons in the special class to filter, 
        or list of indices
    special_color : str or None
        color of the polygons in the special class. If None,
        will assume special_value is a float and get that
        color from the cmap
    special_centroid : bool, default False
        plot centroid to differentiate special class
        polygons
    special_hatch : bool, default False
        hatch special class polygons 
    special_marker_color : str, default `white`
        color for the marker (hatch or centroid) of the 
        special class polygons
        
    bar_plot : float or iterable, default 1.
        value of polygons in the special class to filter, 
        or list of indices

    Returns
    ----------
    ax: matplotlib.Axes
    
    """
    
    #Construct the figure:
    if ax is None: fig, ax = plt.subplots(figsize=(12, 12))

    #Get the colors:
    if cmap is None: cmap=colormaps.get_cmap(cmap_name) #prob_color
    cmap=cropcmap(cmap, min_val=cmap_cropvalue_min, max_val=cmap_cropvalue)
    if special_color is None: special_color=cmap(special_value if not hasattr(special_value, '__iter__') else values_vmax)
    nodata_color='lightgray'
            
    #Get the geodataframe:
    if gdf is None: gdf, _, _ = generate_graph_census('tracts', remove_long_edges=True, remove_zeropop=True)
    gdf['variable'] = values
            
    #Get the base geodataframe:
    if base_gdf is None:
        base_gdf = gdf
    
    bdry_gdf = gpd.GeoDataFrame(geometry=[base_gdf.to_crs(projected_crs).geometry.unary_union], crs=projected_crs)

    #Plot the boundary and a base gdf:
    ax = bdry_gdf.boundary.plot(ax=ax, color='black', zorder=5, linewidth=0.5)
    ax = base_gdf.to_crs(projected_crs).plot(ax=ax, color=nodata_color, linewidth=0, zorder=0)
    
    #Plot the parks:
    if exclude_parks:
        parks_gdf = gpd.read_file(park_shp).to_crs(projected_crs)
        big_parks_gdf = parks_gdf[parks_gdf['typecatego'].isin(['Flagship Park', 'Cemetery'])]
        big_parks_within_gdf = gpd.overlay(big_parks_gdf, base_gdf.to_crs(projected_crs))
        ax = big_parks_within_gdf.plot(color=nodata_color, ax=ax, zorder=2)
    
    #Plot the probabilities:
    colorbar_kwds={'orientation': 'horizontal', 'anchor':(1.,1.), 'panchor':(1.,0.), 'fraction':0.1, 'aspect':30, 'shrink':0.75, 'pad':-0.1}
    #If the colorbar is also the legend for the histogram subplot:
    if inset_colorbar:
        inset_cax = ax.inset_axes([inset_dims[0]-0.1, inset_dims[1]-0.13, 0.1, inset_dims[3]])
        colorbar_kwds.update({'cax':inset_cax, 'pad':0.})
    ax = erase_water(gdf).to_crs(projected_crs).plot(ax=ax, column='variable', vmin=values_vmin, vmax=values_vmax,
                                                     zorder=1,
                                                     cmap=cmap,
                                                     edgecolor='white', linewidth=0,
                                                     legend=True, legend_kwds=colorbar_kwds)
                   
    #Plot the special blocks:
    if special_class:
        #If we have a float, filter:
        if not hasattr(special_value, '__iter__'):
            special_gdf = erase_water(gdf[gdf['variable']==special_value]).to_crs(projected_crs)
        #If we have an iterable, select:
        else:
            special_gdf = erase_water(gdf.iloc[special_value]).to_crs(projected_crs)
        
        #Plot:
        ax = special_gdf.plot(ax=ax, color=special_color, zorder=3, linewidth=0)      
        handle_special = mpatches.Patch(edgecolor='black', facecolor=special_color)
    
        #If we choose to, plot the special centroids:
        if special_centroid:
            special_pt = gpd.GeoDataFrame(geometry=special_gdf.geometry.centroid, crs=projected_crs)
            ax = special_pt.plot(ax=ax, color=special_marker_color, zorder=4, marker='o', markersize=2)
            #Update the handle:
            handle_special = (handle_special, Line2D([],[], linewidth=0, marker='o', markersize=4, color=special_marker_color))
        #If we choose to, plot the report hatches:
        elif special_hatch:
            hatch_gdf = gpd.GeoDataFrame(geometry=[special_gdf.geometry.unary_union], crs=projected_crs)
            ax = hatch_gdf.plot(ax=ax, color=special_color, zorder=4,
                                edgecolor=special_marker_color, linewidth=0.5, hatch='////')
            #Update the handle:
            handle_special.set(fill=None)
            handle_special = (mpatches.Patch(edgecolor=special_marker_color, linewidth=0.5, hatch='////', facecolor=special_color),
                              handle_special)

    #Adjust the label of the color bar:
    _ = ax.text(1-colorbar_kwds['shrink']/2,
                colorbar_kwds['shrink']/colorbar_kwds['aspect']-colorbar_kwds['pad']-0.02,
                values_name,
                ha='center', va='bottom', size=hist_xlabelsize, transform=ax.transAxes, bbox=None)
                   
    #Create the categorical legend:
    handle_nodata = mpatches.Patch(edgecolor='black', facecolor=nodata_color)
    if special_class or legend_nodata:
        if not special_class:
            handles, labels = [handle_nodata], ['No available data']
        elif not legend_nodata:
            handles, labels = [handle_special], [special_label]
        else:
            handles, labels = [handle_nodata, handle_special], ['No available data', special_label]
            
        _ = ax.legend(handles=handles, labels=labels, fontsize=hist_xlabelsize,
                      bbox_to_anchor=(inset_dims[0], inset_dims[1]-0.1), loc='upper left')
        
    #Adjust the axis:
    _ = ax.axis('off')
    _ = ax.set_title(title, fontsize=17)

    #Include a histogram:
    if hist_plot:
        ins_ax = ax.inset_axes(inset_dims)
        axis_label = 'Residents' if hist_weights is not None else ('Census tracts' if unit=='census' else 'Geohashes')
        ins_ax = plot_hist(values,
                           n_bins=hist_bins,
                           range=(values_vmin, values_vmax),
                           fixed_color=hist_fixedcolor,
                           orientation=hist_orientation,
                           weights=hist_weights,
                           special_class=special_class,
                           special_color=special_color,
                           special_hline=special_line,
                           special_centroid=special_centroid,
                           special_hatch=special_hatch,
                           special_marker_color=special_marker_color,
                           cmap=cmap,
                           cmap_name=None,
                           ax=ins_ax,
                           show=False,
                           ylabel=axis_label if hist_orientation=='vertical' else values_name,
                           ylabel_size=hist_ylabelsize,
                           xlabel=axis_label if hist_orientation=='horizontal' else values_name,
                           xlabel_size=hist_xlabelsize,
                           ticks_size=ticks_size)
        
    #Include a bar plot:
    if bar_plot:
        if hist_plot: ins_ax = ax.inset_axes([inset_dims[0], inset_dims[1]-0.26, 0.3, 0.15])
        if not hist_plot: ins_ax = ax.inset_axes(inset_dims)

        if values_name == 'Inferred Probability':
            ins_ax = plot_Aprob_bar(col_to_plot=values,
                                    n_class=bar_n_class, class_upper=bar_class_upper, class_labels=bar_class_labels,
                                    ax=ins_ax, show=False,
                                    cmap=cmap, cmap_name=None, max_val=values_vmax,
                                    reports=special_class,
                                    report_color=special_color,
                                    report_centroid=special_centroid,
                                    report_hatch=special_hatch,
                                    report_marker_color=special_marker_color,
                                    xlabel='Number of census tracts' if unit=='census' else 'Number of Geohashes')
        elif values_name == 'Inferred Report Rate':
            ins_ax = plot_psi_bar(values,
                                  discrete_categories=psibar_categories, weights=psibar_weights, class_labels=psibar_classes,
                                  ax=ins_ax, show=False,
                                  cmap=cmap, cmap_name=None,
                                  draw_hlines=psibar_hlines,
                                  ylabel='Average '+values_name.lower(),
                                  ylabel_size=psibar_ylabelsize,
                                  xlabel_size=psibar_xlabelsize,
                                  ticks_size=ticks_size)
    if show: plt.show()
    if save: fig.savefig(f'{save_dir}map{save_name}.pdf', pad_inches=0.1, bbox_inches='tight', format='pdf')
        
    return ax