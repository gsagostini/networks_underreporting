##########################################################################################################
#This file contains functions to evaluate the allocation of inpsections following our models
##########################################################################################################

import pandas as pd
import numpy as np
from scipy.stats import rankdata

import matplotlib.pyplot as plt
from matplotlib import colormaps as cm

##########################################################################################################

def assess_equity_of_inspections(scores,
                                 training_reports,
                                 capacity=0.1,
                                 weight_ties=True,
                                 model_names=None,
                                 node_weights=None,
                                 node_categories=None,
                                 node_weights_names=None):
    """
    Parameters
    ----------
    scores : list or np.Array
        predicted probabilities of the positive class, either for
        a single model (1D array or flat list) or multiple (2D
        array with shape N_points x N_models)

    training_reports : list or np.Array
        binary observed values of T_i in the training period,
        corresponding to each score which should be ignored
        on the inspection allocation

    capacity : int or float
        how many locations to inspect. If a number between 0 and
        1 is passed, use the corresponding fraction of
        un-reported nodes

    weight_ties : bool
        whether to assign equal fractional inspections to
        nodes that tie at the last position
        
    model_names : list or None
        human-readable model names to use as indices in the
        dataframe. If None, use Model 1, 2, etc.

    node_weights : np.Array or None
        weights of each node, which could be population counts
        for example. If a 2D array is passed, each column
        corresponds to one demographic group.

    node_weights_names : np.Array or None
        groups that each of the weights columns corresponds to,
        if None will be indexed as 1, 2, etc

    node_categories : list or None
        category that each node belongs to (e.g. low-income or
        high-income), which will be groups to count inspections
        according
        
    Returns
    ----------
    equity_per_model : dict
        dictionary where the keys are models and the entries are
        (weighted) counts of inspected nodes according to the
        parameters passed
    """
                                     
    #Convert inputs to arrays:
    if type(scores) == list: scores = np.array(scores)
    if type(training_reports) == list: training_reports = np.array(training_reports)
    N = len(training_reports)
    assert N == scores.shape[0]
    
    #Assign model names:
    if model_names is None:
        model_names = [f'Model {k+1}' for k,_ in enumerate(scores)] if len(scores.shape)==2 else ['Model']
    assert len(model_names) == scores.shape[1] if len(scores.shape)==2 else 1
            
    #Configure the capacity:
    if capacity < 1: capacity = int(capacity*(N-sum(training_reports)))
    assert type(capacity) == int

    #Collect inspection counts:               
    inspection_counts={}
    for model, p in zip(model_names, scores.T):
        model_scores = p[training_reports==0]
        inspection_status = get_inspection_status(scores=model_scores,
                                                  capacity=capacity,
                                                  weight_ties=weight_ties)
        
        #Assign no inspection to reported tracts:
        inspections = np.zeros_like(training_reports)
        inspections[training_reports==0] = inspection_status
        
        #Aggregate inspections with weights:
        inspection_counts[model] = get_inspection_count(inspections,
                                                        weights=node_weights,
                                                        categories=node_categories,
                                                        group_names=node_weights_names)
        
    return inspection_counts

##########################################################################################################

def get_inspection_status(scores,
                          capacity=100,
                          weight_ties=True):
    """
    Determine which nodes get inspected under a particular
        scoring rank.

    Parameters
    ----------
    scores : list
        predicted probabilities used to rank nodes for
        inspection---the higher the value the more priority

    capacity : int
        how many locations to inspect i.e. top-k of the scores
        list will be inspected

    weight_ties : Bool
        whether to consider the ties at the last position
        equally, which results in fractional assignments
        i.e. if 3 nodes tie for the last 2 spots, each 
        node receives 2/3 of inspection
        
    Returns
    ----------
    inspections : np.Array
        array with 0 for nodes not inspected and 1 for nodes
        inspect. could contain fractional values for nodes 
        tied at last position if `weight_ties` is True

    """ 
    #Get the inspection order:
    inspection_order = len(scores) + 1 - rankdata(scores, method='max')
    
    #Inspect those below capacity (i.e. assign 1 to the top k elements):
    inspected = np.heaviside(capacity - inspection_order, 1)

    #This method may incur ties on the last position:
    n_inspected = sum(inspected)
    tie_weight = 1
    if n_inspected > capacity:

        #The ties are entries whose inspection rank is equal to the maximum inspected rank:
        ties = inspection_order == max(inspection_order[inspected == 1])
        n_ties = int(sum(ties))
        n_spots = int(capacity - n_inspected + n_ties)
        
        #Solve ties by weighting tie breaks equally:
        if weight_ties:
            #The weight of each tie should be the number of spots left
            #  divided by the number of ties:
            tie_weight = n_spots/n_ties
            inspected[ties] = tie_weight
            
        #Solve ties at random:
        else:
            tied_inspected = np.random.choice([idx for idx, v in enumerate(ties) if v],
                                               n_ties - n_spots, replace=False)
            inspected[tied_inspected] = 0
  
    return inspected

##########################################################################################################

def get_inspection_count(inspections,
                         weights=None,
                         categories=None,
                         group_names=None):
    """
    Determine how many nodes were inspected, possibly
        by category (or weighting counts). If single weights
        are passed, do a weighted sum of inspected nodes.
        If multiple weights or categories are passed,
        return a dictionary with counts per group.

    Parameters
    ----------
    inspections : list
        assigned inspections per node---mostly 0 or 1 but
        could contain fractional values (ties)

    weights : np.Array or None
        weights of each node, which could be population counts
        for example. If a 2D array is passed, each column
        corresponds to one demographic group.

    group_names : np.Array or None
        groups that each of the weights columns corresponds to,
        if None will be indexed as 1, 2, etc

    categories : list or None
        category that each node belongs to (e.g. low-income or
        high-income), which will be groups to count inspections
        according

        
    Returns
    ----------
    inspections : float or dict
        (weighted) sum of inspections, or dictionary of sums
    """
                             
    #If no demographic is passed, we just want a raw count:
    if weights is None and categories is None:
        count = inspections.sum()

    #If we pass continuous values, we weight the inspections:
    elif weights is not None and categories is None:

        if len(weights.shape)==2:
            count_per_group = (weights*inspections[:, np.newaxis]).sum(axis=0)
            if group_names is None: group_names = [k for k in range(len(weights.shape[1]))]
            count = dict(zip(group_names, count_per_group))
        else:
            count = np.sum(weights*inspections)

    #If we pass categorical values, we count the number of inspections by
    # category and return a dictionary:
    elif weights is None and categories is not None:
        df = pd.DataFrame({'insp': inspections, 'cat': categories})
        count = df.groupby('cat').sum().to_dict()['insp']

    #If we passed both, raise an error:
    else:
        print('Could pass either demographic values or demographic categories but not both!')
        count = None
        
    return count

##########################################################################################################

def get_pop_baselines(training_reports,
                      weights,
                      groups=None):
    """
    Collect the baselines of each subpopulation
        i.e. the total number of residents from
        that group living in un-reported nodes.

    Parameters
    ----------
    training_reports : 1D np.array
        binary observed values of T_i in the training period,
        corresponding to each score which should be ignored
        on the inspection allocation

    weights : 2D np.array
        weights of each node, where each column
        corresponds to one demographic group.

    groups : list
        names of subpopulation groups
        
    Returns
    ----------
    baselines : dict
        dictionary where the keys are groups and the entries
        are the percentage of the unreported population that
        belongs to that group

    """
             
    #If the groups are None:
    if groups is None:
        groups = [k for k in range(len(weights.shape[1]))]
        
    #Get the population on unreported nodes:
    assert type(training_reports) == np.ndarray
    pop_per_group = np.sum(weights[training_reports==0], axis=0)
    pop_total = np.sum(pop_per_group)

    #Baseliens are ratios:
    baseline = dict(zip(groups, pop_per_group/pop_total))

    return baseline
                           
##########################################################################################################

def plot_weighted_equity(scores,
                         training_reports,
                         capacity=0.1,
                         weight_ties=True,
                         model_names=None,
                         node_weights=None,
                         node_weights_names=None,
                         relative_to_baseline=False,
                         plot_baselines=False,
                         groups_to_plot=None,

                         ax=None,
                         colors=None,
                         cmap='YlGnBu',
                         bar_width=0.9,
                         
                         xticks=False,
                         xticks_size=15,
                         xlabel=None,
                         xlabel_size=17,
                         xrange=None,
                         
                         yticks=False,
                         yticks_size=15,
                         ylabel=None,
                         ylabel_size=17,
                         
                         title='Inspections according to subpopulation in tract\n',
                         title_size=16,
                         legend=True,
                         legend_kwds={'loc':'lower right'},
                         show=False):

    """
    Make a bar plot showing residents from each sub-population
        theoretically served by k inspections of unreported
        tracts, ordering by their scores according to each of
        the models.

    Parameters
    ----------
    scores : list or np.Array
        predicted probabilities of the positive class, either for
        a single model (1D array or flat list) or multiple (2D
        array with shape N_points x N_models)

    training_reports : list or np.Array
        binary observed values of T_i in the training period,
        corresponding to each score which should be ignored
        on the inspection allocation

    capacity : int or float
        how many locations to inspect. If a number between 0 and
        1 is passed, use the corresponding fraction of
        un-reported nodes

    weight_ties : bool
        whether to assign equal fractional inspections to
        nodes that tie at the last position
        
    model_names : list or None
        human-readable model names to use as indices in the
        dataframe. If None, use Model 1, 2, etc.

    node_weights : np.Array or None
        weights of each node, which could be population counts
        for example. If a 2D array is passed, each column
        corresponds to one demographic group.

    node_weights_names : np.Array or None
        groups that each of the weights columns corresponds to,
        if None will be indexed as 1, 2, etc

    groups_to_plot : np.Array or None
        sublist of `node_weights_names` that we would like to
        visualize. If None, plot only the first one.

    relative_to_baseline : bool
        whether to return inspected resident counts relative
        to the total subpopulation on un-reported tracts

    plot_baselines : bool
        whether to plot a dashed line corresponding to the
        total subpopulation on unreported tracts. Ignored
        if `relative_to_baseline` is passed.

    ... more plotting related arguments
        
    Returns
    ----------
    ax
    """
                             
    #Convert inputs to arrays:
    if type(scores) == list: scores = np.array(scores)
    if type(training_reports) == list: training_reports = np.array(training_reports)
    N = len(training_reports)
    assert N == scores.shape[0]
    
    #Assign model names:
    if model_names is None:
        model_names = [f'Model {k+1}' for k,_ in enumerate(scores)] if len(scores.shape)==2 else ['Model']
    assert len(model_names) == scores.shape[1] if len(scores.shape)==2 else 1
                             
    #Get the equity dictionary:
    inspection_counts = assess_equity_of_inspections(scores,
                                                     training_reports,
                                                     capacity,
                                                     weight_ties,
                                                     model_names,
                                                     node_weights,
                                                     None,
                                                     node_weights_names)
        
    #Assign colors:
    if colors is None:
        cmap = cm.get_cmap(cmap if cmap is not None else 'viridis')
        colors = cmap(np.linspace(0, 1, len(model_names)))
    assert len(colors) == len(model_names)

    #If we don't pass a list of groups to plot, we plot just the first one:
    if node_weights_names is None:
        if node_weights is None: node_weights_names = ['total']
        else: node_weights_names = [k for k in range(len(node_weights.shape[1]))]
    if groups_to_plot is None: groups_to_plot = node_weights_names[0]

    #Collect baselines:
    if relative_to_baseline or plot_baselines:
        assert node_weights is not None
        baselines = get_pop_baselines(training_reports, node_weights, node_weights_names)

    #Create the ax:
    if ax is None: fig, ax = plt.subplots(figsize=(8, 2.5*len(groups_to_plot)))

    #Iterate over models to plot:
    for model_idx, model in enumerate(model_names):
        
        #The count of inspections for each group are the heights of the bars:
        raw_bar = [inspection_counts[model][group] for group in groups_to_plot]
        
        #Normalize according to capacity or people:
        denominator = capacity if node_weights is None else sum(inspection_counts[model].values())
        bar = np.array([b/denominator for b in raw_bar])

        #Normalize according to baselines:
        if relative_to_baseline: bar = np.array([b/baselines[group]-1 for b, group in zip(bar, groups_to_plot)])
        
        #The y-value of the bars must be adjusted to not overlap:
        y_values = np.array([(len(model_names)+1)*group_idx+model_idx for group_idx in range(len(groups_to_plot))])

        #Plot the bar:
        _ = ax.barh(y_values, 100*bar, height=bar_width, label=model, color=colors[model_idx], alpha=0.9)

    #Get the y ticks:
    if yticks:
        _ = ax.set_yticks([(len(model_names)+1)*k+(len(model_names)-1)/2 for k in range(len(groups_to_plot))],
                          [group.capitalize() for group in groups_to_plot], fontsize=yticks_size)
    else:
        _ = ax.set_yticks([],[])

    #Label in the y axis:
    if ylabel is not None:
        _ = ax.set_ylabel(ylabel, ha='left', y=1.05, rotation=0, labelpad=0, fontsize=ylabel_size)

    _ = ax.set_ylim(-bar_width, len(model_names)+bar_width-1)
        
    #Add the baseline (% of unreported tracts from that group):
    if plot_baselines and not relative_to_baseline:
        x = [100*baselines[group] for group in groups_to_plot]
        _ = ax.vlines(x=x,
                      ymin=ax.get_ylim()[0],
                      ymax=ax.get_ylim()[1],
                      linestyle='--', color='k',
                      label=None)
        
    #Label in the x axis:
    if xlabel and (type(xlabel)!=str): xlabel = f"% of all served {'census tracts' if node_weights is None else 'residents'}"
    _ = ax.set_xlabel(xlabel, fontsize=xlabel_size, ha='right', x=1.)
    if xrange is not None: _ = ax.set_xlim(xrange)
                             
    #Configure axis:
    _ = ax.set_title(title, fontsize=title_size)
    _ = ax.spines[['right', 'top']].set_visible(False)
    if relative_to_baseline: _ = ax.spines['left'].set_position(('data', 0))

                             
    if legend: _ = ax.legend(**legend_kwds)
    if show: plt.show()    
        
    return ax

##########################################################################################################

def plot_line_equity(scores,
                     training_reports,
                     capacities=(1,201),
                     weight_ties=True,
                     model_names=None,
                     colors=None,
                     cmap='YlGnBu',
                     node_weights=None,
                     node_weights_names=None,
                     groups_to_plot=None,
                     relative_to_baseline=False,
                     plot_baselines=True,
                     ax=None,
                     yrange=None,
                     ylabel='% of all inspected residents',
                     ylabel_size=13,
                     xlabel='Number of inspected tracts',
                     xlabel_size=13,
                     title=None,
                     title_size=16,
                     show=False,
                     legend=True,
                     legend_kwds={'loc':'lower right'}):

    """
    Make a line plot showing residents from each sub-population
        theoretically served by k inspections of unreported
        tracts for multiple k, ordering by their scores according
        to each of the models.

    Parameters
    ----------
    scores : list or np.Array
        predicted probabilities of the positive class, either for
        a single model (1D array or flat list) or multiple (2D
        array with shape N_points x N_models)

    training_reports : list or np.Array
        binary observed values of T_i in the training period,
        corresponding to each score which should be ignored
        on the inspection allocation

    capacities : tuple
        minimum and maximum capacities

    weight_ties : bool
        whether to assign equal fractional inspections to
        nodes that tie at the last position
        
    model_names : list or None
        human-readable model names to use as indices in the
        dataframe. If None, use Model 1, 2, etc.

    node_weights : np.Array or None
        weights of each node, which could be population counts
        for example. If a 2D array is passed, each column
        corresponds to one demographic group.

    node_weights_names : np.Array or None
        groups that each of the weights columns corresponds to,
        if None will be indexed as 1, 2, etc

    groups_to_plot : np.Array or None
        sublist of `node_weights_names` that we would like to
        visualize. If None, plot only the first one.

    relative_to_baseline : bool
        whether to return inspected resident counts relative
        to the total subpopulation on un-reported tracts

    plot_baselines : bool
        whether to plot a dashed line corresponding to the
        total subpopulation on unreported tracts. Ignored
        if `relative_to_baseline` is passed.

    ... more plotting related arguments
        
    Returns
    ----------
    ax
    """
    #Convert inputs to arrays:
    if type(scores) == list: scores = np.array(scores)
    if type(training_reports) == list: training_reports = np.array(training_reports)
    N = len(training_reports)
    assert N == scores.shape[0]
    
    #Assign model names:
    if model_names is None:
        model_names = [f'Model {k+1}' for k,_ in enumerate(scores)] if len(scores.shape)==2 else ['Model']
    assert len(model_names) == scores.shape[1] if len(scores.shape)==2 else 1

    #Assign colors:
    if colors is None:
        cmap = cm.get_cmap(cmap if cmap is not None else 'viridis')
        colors = cmap(np.linspace(0, 1, len(model_names)))
    assert len(colors) == len(model_names)

    #If we don't pass a list of groups to plot, we plot just the first one:
    if node_weights_names is None:
        if node_weights is None: node_weights_names = ['total']
        else: node_weights_names = [k for k in range(len(node_weights.shape[1]))]
    if groups_to_plot is None: groups_to_plot = node_weights_names[0]

    #Collect baselines:
    if relative_to_baseline or plot_baselines:
        assert node_weights is not None
        baselines = get_pop_baselines(training_reports, node_weights, node_weights_names)

    #Create the ax:
    if ax is None: fig, ax = plt.subplots(figsize=(5, 3*len(groups_to_plot)), nrows=len(groups_to_plot))
    if type(ax) == plt.Axes: ax = [ax]
        
    #Iterate over groups:
    capacities_arr = range(*capacities)
    for group, group_ax in zip(groups_to_plot, ax):
        
        #Iterate over possible capacities:
        inspection_counts_dict = {model:[] for model in model_names}
        for capacity in capacities_arr:

            #Get the equity dictionary:
            inspection_counts = assess_equity_of_inspections(scores,
                                                             training_reports,
                                                             capacity,
                                                             weight_ties,
                                                             model_names,
                                                             node_weights,
                                                             None,
                                                             node_weights_names)
            #Normalize the counts on each model:
            for model in model_names:
                
                raw_count = inspection_counts[model][group]
                
                #Normalize per total population inspected:
                denominator = capacity if node_weights is None else sum(inspection_counts[model].values())
                normalized_count = raw_count/denominator
                
                #Normalize according to baselines:
                if relative_to_baseline:
                    normalized_count = normalized_count/baselines[group]-1

                inspection_counts_dict[model].append(100*normalized_count)

        #Plot:
        for model_idx, model in enumerate(model_names):
            _ = group_ax.plot(capacities_arr, inspection_counts_dict[model], label=model, color=colors[model_idx])
            
        #Include baselines:
        if plot_baselines and not relative_to_baseline:
            _ = group_ax.axhline(y=100*baselines[group],
                                 xmin=0,
                                 xmax=capacities[1],
                                 linestyle='--', color='k', label=None, zorder=10)

        #Customize axis:
        if yrange is not None: _ = group_ax.set_ylim(*yrange)
        _ = group_ax.set_ylabel(ylabel, fontsize=ylabel_size)
        _ = group_ax.set_xlim(*capacities)
        _ = group_ax.set_xlabel(xlabel, fontsize=xlabel_size, ha='right', x=1.)
        
        if title is not False: _ = group_ax.set_title(title, fontsize=title_size)

        _ = group_ax.spines[['right', 'top']].set_visible(False)
        if relative_to_baseline: _=group_ax.axhline(y=0, linestyle='-', color='k', zorder=10, linewidth=0.8)#_ = group_ax.spines['bottom'].set_position(('data', 0))
            
        if legend: _ = group_ax.legend(**legend_kwds)

    if len(ax) == 1: ax = ax[0]
    if show: plt.show()
    return ax