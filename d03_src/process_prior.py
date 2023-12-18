############################################################################
# Functions to study the priors of our model:
############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import sys
sys.path.append('../d03_src/')
import vars

############################################################################

def get_means_and_keys(dictionary):
    A_means = {params:A.mean(axis=1) for params, A in dictionary.items()}
    possible_theta_0, possible_theta_1 = zip(*list(A_means.keys()))
    possible_theta_0 = list(set(possible_theta_0))
    possible_theta_1 = list(set(possible_theta_1))
    possible_theta_0.sort()
    possible_theta_1.sort()
    return possible_theta_0, possible_theta_1, A_means

############################################################################

def generate_plots_dist_per_graph(dictionary, possible_theta_0=None, possible_theta_1=None, leg_all=False):

    #Get the keys and means of the distributions:
    keys_possible_theta_0, keys_possible_theta_1, A_means = get_means_and_keys(dictionary)
    if possible_theta_0 is None: possible_theta_0 = keys_possible_theta_0
    if possible_theta_1 is None: possible_theta_1 = keys_possible_theta_1

    #Create the figure:
    fig, Axes = plt.subplots(figsize=(10, 5*len(possible_theta_1)), nrows=len(possible_theta_1))

    #Fill one axis at a time:
    for ax_idx, theta_1 in enumerate(possible_theta_1):
        
        ax = Axes[ax_idx]
        
        #Plot the distribution ofor each theta_0:
        for theta_0 in possible_theta_0:
            scaled_means = A_means[(theta_0, theta_1)]/2 + 0.5
            sns.kdeplot(data=scaled_means, ax=ax, label=theta_0)

        #Set the title:
        _ = ax.set_title(fr'Distributions for $\theta_1 = {theta_1}$', size=20)

        #Configure the legend:
        if leg_all or ax_idx == len(possible_theta_1)-1:
            handles, labels = ax.get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: float(t[0])))
            leg = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=8, frameon=False)
            _ = leg.set_title(r'$\theta_0$',prop={'size':20})

        #Set axes limits:
        _ = ax.set_xlim(0, 1)
        
    return Axes
    
############################################################################

def plot_heatmap(dictionary, full_plot=True, cmap='BrBG', fig_len=8, ax=None, title_l2='', show=True, return_hm=False, annotate=False):

    #Get the keys and means of the distributions:
    possible_theta0, possible_theta1, A_means = get_means_and_keys(dictionary)

    #Create the figure with a colorbar divider if full_plot:
    if ax is None: fig, ax = plt.subplots(figsize=(fig_len, fig_len))
    if full_plot: divider = make_axes_locatable(ax)
    if full_plot: cax = divider.append_axes('right', size='5%', pad=0.5)

    #Find the heatmap array:
    heatmap = np.array([[np.median(A_means[(theta0, theta1)]) for theta1 in possible_theta1] for theta0 in possible_theta0])
    heatmap = heatmap*50 + 50

    #Plot:
    hm = ax.imshow(heatmap, cmap=cmap, vmin=0, vmax=100, aspect='equal', origin='lower')

    #Legend:
    if full_plot: fig.colorbar(hm, cax=cax, orientation='vertical')

    #Axes labels:
    ax.set_yticks(range(len(possible_theta0))[::2], labels=possible_theta0[::2], rotation=0)
    ax.set_ylabel(r'$\theta_0$', size=12, rotation=0, labelpad=10)
    ax.set_xticks(range(len(possible_theta1))[::2], labels=possible_theta1[::2], rotation=0)
    ax.set_xlabel(r'$\theta_1$', size=12, rotation=0)

    #Title:
    if full_plot: ax.set_title(f"Average % of events according to true parameters\n{title_l2}", fontsize=20)

    #Add values annotations:
    if annotate:
        for i, pct_list in enumerate(heatmap):
            for j, pct in enumerate(pct_list):
                text = ax.text(j, i, int(pct), ha="center", va="center", color="black")
        
    if show: plt.show()

    return ax if not return_hm else ax, hm

#####################################################
# Checking distribution per prior:

from scipy.stats import norm
from sklearn.neighbors import KernelDensity

def p_theta(theta, theta_mu=[0., 0.], theta_sigma=[1., 1.]):
    return norm.pdf(theta[0], loc=theta_mu[0], scale=theta_sigma[0])*norm.pdf(theta[1], loc=theta_mu[1], scale=theta_sigma[1])

def get_denominator(possible_theta_values, theta_mu=[0., 0.], theta_sigma=[1., 1.]):
    all_values = [p_theta(theta, theta_mu, theta_sigma) for theta in possible_theta_values]
    return np.sum(all_values)
    
def prob_mean_A(mean_A, mean_A_dict, theta_mu, theta_sigma, denominator=None, kd_dict=None):

    #Collect the possible values:
    possible_theta_values = list(mean_A_dict.keys())

    #Collect the kernel estimators:
    if kd_dict is None: kd_dict = {theta: KernelDensity(kernel='gaussian', bandwidth=0.015).fit((means/0.5 + 0.5).reshape(-1,1)) for theta, means in mean_A_dict.items()}

    #Integrate:
    integrand = [np.exp(kd_dict[theta].score_samples([[mean_A]]))*p_theta(theta, theta_mu, theta_sigma) for theta in possible_theta_values]
    unweighted_integral = np.sum(integrand)

    #Normalize:
    if denominator is None: denominator = get_denominator(possible_theta_values, theta_mu, theta_sigma)
    integral = unweighted_integral / denominator
    
    return integral

def estimate_prob_mean_A(A_dictionary, prior_mu, prior_sigma, possible_A=np.linspace(0, 1, 101), bw=0.015, verbose=False):
    
    _, _, A_mean_dict = get_means_and_keys(A_dictionary)
    possible_theta_values = list(A_mean_dict.keys())
    denominator = get_denominator(possible_theta_values, theta_mu=prior_mu, theta_sigma=prior_sigma)
    
    if verbose:
        print('mu:', prior_mu)
        print('sigma:', prior_sigma)
        print('normalization:', denominator)

    kd_dict = {theta: KernelDensity(kernel='gaussian', bandwidth=bw).fit((means/0.5 + 0.5).reshape(-1,1)) for theta, means in A_mean_dict.items()}
    prob_arr = [prob_mean_A(mean, A_mean_dict, prior_mu, prior_sigma, denominator, kd_dict) for mean in possible_A]

    return possible_A, prob_arr