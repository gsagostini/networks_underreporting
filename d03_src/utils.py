import os
import sys
import threading
import logging
# silence logger, there are better ways to do this
# see PyStan docs
logging.getLogger("stan").propagate=False
import numpy as np
import networkx as nx
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.special import logsumexp, expit
from scipy.spatial import distance
from copy import deepcopy

import sys
sys.path.append('../d03_src/')
from model_SVEA import get_density, sample_SVE
from vars import _covariates, _text_width, _column_width, _param_description

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.utils import resample

def param_name(parameter):
    param_latex_string = f'{parameter[:-1]}_{parameter[-1]}'
    param_description = _param_description[parameter]
    return param_description + fr' ($\{param_latex_string}$)'

def set_figsize(width='column', fraction=1, height_ratio=(5**.5 - 1)/2, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'column':
        width_pt = _column_width
    elif width == 'text':
        width_pt = _text_width
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * height_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def cropcmap(cmap, min_val=None, max_val=None, precision=512):

    colors = cmap(np.linspace(0, 1, precision))

    #Crop above:
    if max_val is not None:
        plateau_length = int((1/max_val-1)*precision)
        plateau_colors = np.vstack([colors[-1]]*plateau_length)
        colors = np.vstack([colors, plateau_colors])

    #Crop below:
    if min_val is not None:
        plateau_length = int((1/(1-min_val)-1)*precision)
        plateau_colors = np.vstack([colors[0]]*plateau_length)
        colors = np.vstack([plateau_colors, colors])
    
    return ListedColormap(colors)

def boot(array, n_bootstrap=10_000, confidence_level=0.95):
    
    N = len(array)
    vals = []
    
    for iteration in range(n_bootstrap):
        
        #Get a bootstraped sample:
        boot_idx = resample(range(N), replace=True)
        boot_vals = array[boot_idx]

        #Calculate statistic:
        vals.append(boot_vals.mean())
        
    boot_array = np.random.choice(array, size=n_bootstrap, replace=True)

    #Compute the CI ignoring nan values
    estimate = np.mean(vals)
    LB, UB = np.nanpercentile(vals,
                              [50*(1-confidence_level), 50*(1+confidence_level)])
    
    return estimate, LB, UB

def get_delta(row, metric_df, bootstrap=False, n_bootstrap=10_000, confidence_level=0.95):

    #Compute the differences:
    values_A = metric_df[row['Model A']].to_numpy()
    values_B = metric_df[row['Model B']].to_numpy()

    #Whether or not we bootstrap these simulations:
    if bootstrap:
        N = len(values_A)
        delta = []
        for iteration in range(n_bootstrap):
            
            #Get a bootstraped sample:
            boot_idx = resample(range(N), replace=True)
            boot_vals_A = values_A[boot_idx]
            boot_vals_B = values_B[boot_idx]

            #Calculate statistic:
            boot_deltas = boot_vals_A - boot_vals_B 
            delta.append(boot_deltas.mean())

    else:
        delta = values_A - values_B

    #Compute the estimate (mean) and CI:
    estimate = np.mean(delta)
    LB, UB = np.nanpercentile(delta,
                              [50*(1-confidence_level), 50*(1+confidence_level)])

    return estimate, LB, UB
    
def latex(variable):
    if 'alpha' in variable: variable = rf'$\alpha_{variable[5:]}$'
    if variable == 'theta0': variable = r'$\theta_0$'
    if variable == 'theta1': variable = r'$\theta_1$'
    if variable == 'psi': variable = r'$\alpha$'
    if variable == 'A_mean': variable = r'mean($\vec{A}$)'
    if variable == 'r_hat': variable = r'$\hat{r}$'
    
    return variable

def annotate_params(variable, covariates=_covariates):

    annotation = variable
 
    if variable == 'theta0': annotation = r'prevalence'
    if variable == 'theta1': annotation = r'spatial correlation'
    if variable == 'psi': annotation = r'report rate'
    if variable == 'A_mean': annotation = r'ground truth mean'
    
    if 'alpha' in variable:
        idx = int(variable[5:])
        if idx == 0:
            annotation = r'intercept'
        else:
            cov=covariates[idx-1]
            if cov == 'log_population': annotation = r'population'
            if cov == 'income_median': annotation = r'income'
            if cov == 'education_bachelors_pct': annotation = r'education'
            if cov == 'race_white_pct': annotation = r'race'
            if cov == 'age_median': annotation = r'age'
            if cov == 'households_owneroccupied_pct': annotation = r'homeownership'

    return annotation

def drain_pipe(captured_stdout, stdout_pipe):
    while True:
        data = os.read(stdout_pipe[0], 1024)
        if not data:
            break
        captured_stdout += data


def capture_output(function, *args, **kwargs):
    """
    https://stackoverflow.com/questions/24277488/in-python-how-to-capture-the-stdout-from-a-c-shared-library-to-a-variable
    """
    stdout_fileno = sys.stdout.fileno()
    stdout_save = os.dup(stdout_fileno)
    stdout_pipe = os.pipe()
    os.dup2(stdout_pipe[1], stdout_fileno)
    os.close(stdout_pipe[1])

    captured_stdout = b''

    t = threading.Thread(target=lambda:drain_pipe(captured_stdout, stdout_pipe))
    t.start()
    # run user function
    result = function(*args, **kwargs)
    os.close(stdout_fileno)
    t.join()
    os.close(stdout_pipe[0])
    os.dup2(stdout_save, stdout_fileno)
    os.close(stdout_save)
    return result, captured_stdout.decode("utf-8")

def adapt_step_size(current_step,
                    acceptance_rate,
                    bounds=(0.25, 0.6), 
                    multiplier=0.1):
    
    if acceptance_rate <= bounds[0]:
        proposal_sigma = current_step*(1-multiplier)
    elif acceptance_rate >= bounds[1]:
        proposal_sigma = current_step*(1+multiplier)
    else:
        proposal_sigma = current_step
        
    return proposal_sigma


############################################################################
# EXACT LIKELIHOOD:

def exact_density(graph,
                  A,
                  theta,
                  do_log=True):
    """
    Compute the exact density f(A|theta) for our Ising distribution,
      which is intractable for large graphs.
      
    Parameters
    ----------
    graph : Networkx.Graph
        graph object whose nodes correspond to spatial
        locations, must be labeled 0...N-1
    A : np.array
        true values for the hidden state of each node,
        either +1 or -1 (both cases must appear)
    theta: list
        list of two theta values
    do_log : Boolean
        whether to return probabilities in logspace

    Returns
    ----------
    density: float
    """
    
    N = graph.order()
    graph_matrix=nx.adjacency_matrix(graph)
    
    assert len(A) == N
    assert all([a in [-1, 1] for a in A])
    
    #The numerator is the (un-normalized) density of the given assignment:
    numerator = get_density(theta,
                            graph_matrix=graph_matrix,
                            A=A,
                            log=do_log)
    
    #For the denominator, we must compute the un-normalized density of
    # all possible assignments for the given thetas:    
    all_assignments = np.array(list(itertools.product([-1, 1], repeat=N)))
    assert len(all_assignments) == 2 ** N
    
    def get_density_for_assignment(a):
        return get_density(theta, graph_matrix, a, log=do_log)
    all_densities = np.apply_along_axis(get_density_for_assignment, 1, all_assignments)
    
    if do_log:
        return numerator - logsumexp(all_densities)
    
    else:
        return numerator/sum(all_densities)
        
##################################################################################################################################
def add_trivial_inspections(df, graph, do_Aprob=True):
    A = nx.adjacency_matrix(graph, nodelist=df.index.values)
    T = df.T_train.values
    w = A.dot(T)
    df['inf_Aprob'] = np.maximum(w/A.sum(axis=1), T) if do_Aprob else w/A.sum(axis=1)

    return df

def add_GP_inspections(df, gdf, scale_by=1000, loc=1):

    #Get the array of locations and scale:
    X = np.array([gdf.centroid.x.values, gdf.centroid.y.values]).T
    X_scale = (X - np.mean(X, axis=0))/scale_by

    #Define the Gaussian process
    kernel = RBF(loc)
    GP = GaussianProcessRegressor(kernel=kernel, normalize_y=False, optimizer=None)

    #Fit to training data:
    X_train_idx = df[df['T_train'] == 1].index.to_numpy()
    GP_fitted = GP.fit(X_scale[X_train_idx], [1]*len(X_train_idx))

    #Predict:
    df['inf_Aprob'] = np.clip(GP_fitted.predict(X_scale), 0, 1)

    return df
    
############################################################################
#QUANTILES

def q5(x):
    return x.quantile(0.05)
def q25(x):
    return x.quantile(0.25)
def q75(x):
    return x.quantile(0.75)
def q95(x):
    return x.quantile(0.95)

############################################################################

def sample_exact(graph, A,
                 n_samples=50_000,
                 theta_0_bounds=(-2, 2),
                 theta_1_bounds=(-2, 2),
                 n_thetas=100,
                 do_log=True):
    
    #Find candidates for theta:
    theta_0_vals = np.linspace(theta_0_bounds[0], theta_0_bounds[1], n_thetas)
    theta_1_vals = np.linspace(theta_1_bounds[0], theta_1_bounds[1], n_thetas)
    thetas = np.array([(theta_0, theta_1)
                       for theta_0 in theta_0_vals
                       for theta_1 in theta_1_vals])
    
    #Get the exact likelihood (or log-likelihood)
    def get_density_for_theta(theta):
        return exact_density(graph, A, theta, do_log=do_log)
    pr_a_given_thetas = np.apply_along_axis(get_density_for_theta, 1, thetas)
    if do_log:
        pr_a_given_thetas = np.exp(pr_a_given_thetas)
    
    #Get a sample:
    results = pd.DataFrame(np.append(thetas, pr_a_given_thetas.reshape((-1, 1)), 1),
                           columns=['theta0', 'theta1', 'Pr(A|theta)'])
    results['weight'] = results['Pr(A|theta)'] / results['Pr(A|theta)'].min()
    exact_samples = results.sample(n=n_samples, replace=True, weights=results['weight'])[['theta0', 'theta1']].values
    
    return exact_samples


#COMPARE TO EXACT SAMPLING

def verify_SVAE_with_exact_sampling(graph, A,
                                    plot_distributions=False,
                                    return_samples=False,
                                    n_samples=50000,
                                    do_log=True):
    
    #Get the exact likelihood:
    exact_samples = sample_exact(graph, A, n_samples=n_samples)
    print(exact_samples)
    exact_samples_df = pd.DataFrame(exact_samples, columns=['theta0', 'theta1'])
    exact_summary = exact_samples_df.agg(['mean', 'std', q5, q25, 'median', q75, q95]).transpose()
    
    #Approximate with SVAE:
    SVAE_theta_samples = sample_SVE(graph, np.array(A), n_iterations=n_samples)
    SVAE_theta_samples_df = pd.DataFrame(SVAE_theta_samples, columns=['theta_0', 'theta_1'])
    SVAE_summary = SVAE_theta_samples_df.iloc[500:].agg(['mean', 'std', q5, q25, 'median', q75, q95]).transpose()
    
    #If we want to plot:
    if plot_distributions:
        print('\n')
        fig, Ax = plt.subplots(figsize=(10,10), nrows=2, sharex=True)
        SVAE_theta_samples_df.hist(ax=Ax, label='SVAE', alpha=0.7, density=True, range=(-2, 2), bins=21)
        exact_samples_df[['theta0','theta1']].reset_index(drop=True).hist(ax=Ax, label='exact', alpha=0.7, density=True, range=(-2, 2), bins=21)
        _ = [ax.legend() for ax in Ax.flatten()]
        plt.show()
    
    if return_samples:
        return exact_summary, SVAE_summary, exact_samples_df.reset_index(drop=True), SVAE_theta_samples_df
    else:
        return exact_summary, SVAE_summary

#WEIGHTED MEDIAN
def weighted_median(values, weights):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, 0.5 * c[-1])]]

############################################################################

def add_psi(coefficients, covariates, intercept=True):
    """
    Given the alpha coefficients and an array of node
        specific covariates, compute the reporting
        raters psi_i for all nodes in the network.
    """
    #In case we need to consider the first coeff. as alpha_0
    if intercept:
        psi = expit(coefficients[0] + np.sum(coefficients[1:]*covariates, axis=1))
    #In pooled models, we do not have an intercept:
    else:
        psi = expit(np.sum(coefficients*covariates, axis=1))
    return psi

