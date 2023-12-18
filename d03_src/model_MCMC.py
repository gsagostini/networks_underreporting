############################################################################
# Functions of the top-level Markov Chain to sample from our model
############################################################################

from tqdm import tqdm
import numpy as np
import copy
from scipy.special import expit, logit

import sys
sys.path.append('../d03_src/')
from model_SVEA import sample_SVE, adapt_SVE_stepsize
from model_regression import reg_sample_alphas

############################################################################
# MARKOV CHAIN MONTE CARLO SAMPLING

def run_MCMC(graph,
             covariates=None,
             fixed_T=False,
             fixed_params=False,
             T=None,
             params=None,
             MCMC_iterations=5_000,
             MCMC_burnin=500,
             MCMC_return_burnin=True,
             MCMC_thinfrac=1.,
             MCMC_verbose=False,
             params_sample_start=None,
             params_sample_dict=None,
             debug=False, 
             return_stepsize=False):
    """
    Runs a Markov Chain MonteCarlo to sample either the variables (A and T) or the
      parameters (alphas and thetas) of our model. Must give either true parameters or T in
      order to sample/infer the variable A and the other.
      
    Model parameter dictionaries keys are `alpha0`, ... `alphaM`, `theta0`, and `theta1`.
    
    Parameters
    ----------
    graph : Networkx.Graph
        graph object whose nodes correspond to spatial
        locations, must be labeled 0...N-1

    covariates : np.Array of shape N x M or None
        demographic covariates per node that will be used
        in the regression to determine reporting rate. If None,
        report rate is fixed across all nodes
        
    fixed_T : Bool, default False
        whether to consider T observed and thus sample only
        parameters. exactly one of `fixed_T` or `fixed_params`
        must be True       
    fixed_params : Bool, default False
        whether to consider parameters known and thus sample
        only T. exactly one of `fixed_T` or `fixed_params` must
        be True        
    T : None or np.array, default None
        if `fixed_T` is True, the value of observed T variables
    params : None or dict, default None
        if `fixed_params` is True, the value of known parameters
        
    MCMC_iterations : int, default 5000
        total number of iterations in the chain
    MCMC_burnin : int, default 500
        number of initial iterations to be discarded
    MCMC_return_burnin : Bool, default True
        whether to also return the burn-in samples
    MCMC_thinfrac : float in (0, 1), default 0.4
        proportions of iterations in chain to keep,
        as to reduce correlation
    MCMC_verbose : Bool, default False
        whether to print mode and status
        
    params_sample_start: None or dict
        if `fixed_T` is True, dictionary of starting parameter
        values. if None, starting parameters are random
    params_sample_dict : dict
        dictionary with arguments for sampling parameters. Possible keys:
            
            psi_method : 'Reg' or 'Beta', default 'Reg'
               method to use for sampling report rates---if Beta, covariates
               will be ignored and report rate will be constant
               
            psi_Beta_priormean : float in (0, 1), default .5
                mean to use in Beta prior of psi.
            psi_Beta_priorstrength : float in (0, inf), default 2.0
                strength to use in Beta prior of psi.

            psi_Reg_warmup : int, default 50
                number of warmup samples in Bayesian logistic regression
            psi_Reg_model : pymc.Model or None, default None
                Bayesian logistic regression model, usually starts as 
                None and gets built in the first iteration
            psi_Reg_priormean : list of length M or None, default None
                mean to use in normal priors of regression coefficients.
                if None, default to 0.
            psi_Reg_priorsd : list of length M or None, default None
                mean to use in normal priors of regression coefficients.
                if None, default to 10.

            theta_method : 'Exact' or 'SVE', default 'SVE'
               method to use for sampling theta
            
            theta_Exact_theta0bounds : tuple, default (-2, 2),
            theta_Exact_theta1bounds : tuple, default (-2, 2),
            theta_Exact_nthetas : int, default 200
            
            theta_SVE_warmup : int, default 100
            theta_SVE_auxiliary : int, default 50
            theta_SVE_proposalsigma : float, default 0.2
            theta_SVE_adaptive : Bool, default True
            theta_SVE_adaptiveiter : int, default 50
            theta_SVE_adaptivebounds : tuple, default (0.25, 0.6)
            theta_SVE_adaptiverate : float, default 0.15
            theta_SVE_start : list, default [0., 0.3]
            theta_SVE_priormean : list, default [0., 0.3]
            theta_SVE_priorsigma : list, default [0.2, 0.1]

    debug : Bool or list
        whether to return debug statistics. If a list is passed, specific debugging variables
        are stored. Possible values are `stepsize`, `SVE`, `SVE_simple`. if True, all
        debug variables are stored

    Returns
    ----------
    sampled_A: list
        list of samples on the variable A. Each element is a list of size N, nodes in default
        ordering. The variable is always sampled.    
    sampled_T: list
        list of samples on the variable T. Each element is a list of size N, nodes in default
        ordering. The T variable is sampled if and only if `fixed_params` is `True`.
    sampled_params: list
        list of samples of the parameters. Each element is a dictionary. The parameters are
        sampled if and only if `fixed_T` is `True`.

    If debug != False, also returns debug statistics
    
    Returned lists are either empty (variable not sampled) or have the same length. List
      length varies according to the number of MCMC iterations, the number of burn-in
      iterations, and thinning fraction.
    """
    
    #Must have specified T xor params fixed:
    assert fixed_T^fixed_params
    
    #Initialization:
    N, M, A, T, params = initialize_MCMC(graph, covariates,
                                         fixed_T, fixed_params,
                                         T, params,
                                         params_sample_start,
                                         MCMC_verbose)
    
    #Filling in hyperparameters for sampling:
    params_sample_dict = add_default_param_HP(params_sample_dict)
    
    #Keep lists:
    sampled_A, sampled_T, sampled_params = [], [], []

    #Auxiliary lists:
    debug_variables, acceptance_rates, stepsize_list = [], [], []
    
    #Iterate:
    for iteration in tqdm(range(MCMC_iterations)) if MCMC_verbose else range(MCMC_iterations):
        
        #Assert the PU assumption:
        assert 0 not in A + T
        
        #Update A and possibly T:
        A, T = update_AT(A, T, params, graph, fixed_T)
        
        #Update parameters:
        if not fixed_params:
            params, debug_args = sample_params(A, T, params, graph,
                                               covariates=covariates,
                                               HP=params_sample_dict,
                                               debug=debug)
            acceptance_rates.append(debug_args['theta_SVE_acceptance'])
            
        #If on burn-in period, update chain parameters:
        if iteration < MCMC_burnin:
            
            #For adaptive stepsize routines:
            params_sample_dict, acceptance_list = adapt_SVE_stepsize(params_sample_dict,
                                                                     acceptance_rates,
                                                                     iteration)
            #If we are logging burn-in samples:
            if MCMC_return_burnin:
                sampled_A.append(copy.deepcopy(A))
                sampled_T.append(copy.deepcopy(T))
                sampled_params.append(copy.deepcopy(params))
                if debug: debug_list.append(debug_args)
                if return_stepsize: stepsize_list.append(params_sample_dict['theta_SVE_proposalsigma'])
            
        #If on the post burn-in period, and chain is thinned:
        elif iteration >= MCMC_burnin and np.random.random() <= MCMC_thinfrac:
            sampled_A.append(copy.deepcopy(A))
            sampled_T.append(copy.deepcopy(T))
            sampled_params.append(copy.deepcopy(params))
            if debug: debug_list.append(debug_args)
    
    #Check whether we need to return the debug arguments:
    if debug:
        return sampled_A, sampled_T, sampled_params, debug_list
    else:
        if return_stepsize:
            return sampled_A, sampled_T, sampled_params, stepsize_list
        else:
            return sampled_A, sampled_T, sampled_params

############################################################################
# INITIALIZATION

def initialize_MCMC(graph, covariates=None,
                    fixed_T=False, fixed_params=False,
                    T=None, params=None,
                    starting_params=None,
                    verbose=False):
    """
    Initializes the Markov Chain Monte Carlo, selecting values for
      A, T, and the parameters if initial values are not given.
      
    Model parameter dictionaries keys are `alpha0`...`alphaM`, `theta0`, and `theta1`.
    
    Parameters
    ----------
    graph : Networkx.Graph
        graph object whose nodes correspond to spatial
        locations, must be labeled 0...N-1
    covariates : np.Array of shape N x M or None
        demographic covariates per node that will be used
        in the regression to determine reporting rate. If None,
        report rate is fixed across all nodes
    fixed_T : Bool, default False
        whether to consider T observed and thus sample only
        parameters. exactly one of `fixed_T` or `fixed_params`
        must be True
    fixed_params : Bool, default False
        whether to consider parameters known and thus sample
        only T. exactly one of `fixed_T` or `fixed_params` must
        be True
    T : None or np.array, default None
        if `fixed_T` is True, the value of observed T variables
    params : None or dict, default None
        if `fixed_params` is True, the value of known parameters
    starting_params : None or dict, default None
        dictionary of starting parameter values. if None, initial
        parameters are random. ignored if `fixed_params` is True
        
    Returns
    ----------
    N: int
        number of nodes in the graph
    M: int
        number of covariates
    initial_A: np.array
        1D array of length N with entries -1 or 1 corresponding
        to whether an incident occurred at a given node
    initial_T: np.array
        1D array of length N with entries 0 or 1 corresponding
        to whether an incident was reported at a given node
    initial_params: dict
        dictionary with keys `alpha0`...`alphaM`, `theta0`, and `theta1`
        
    A and T will satisfy the PU assumption: if A=-1 then
      the corresponding T=0.
    """
    #Initialize common values:
    A, N = initialize_AN(graph)
    M = 0 if covariates is None else len(np.array(covariates).T)
    
    #Initialize T:
    if fixed_params:
        if 'psi' not in params: params['psi'] = compute_psi(params, covariates, M)
        T = initialize_T(params, N, A)
    A[T == 1] = 1
    
    #Initialize parameters:
    if fixed_T: params = initialize_params(starting_params, covariates, M)
        
    if verbose:
        fixed_variable = 'T' if fixed_T else 'parameters'
        sampled_variable = 'T' if fixed_params else 'parameters'
        print(f'Initializing the chain with fixed {fixed_variable}')
        print(f'Will sample A and {sampled_variable}')
        print(f'{M} covariates passed')
        param_status = 'True' if fixed_params else 'Initialized'
        param_string = ', '.join([f'{k[0]} = {k[1]:.2f}' for k in params.items() if k[0] != 'psi'])
        print(f'{param_status} parameters: {param_string}')
        print('\n')
        
    return N, M, A, T, params

def initialize_AN(graph):
    N = int(graph.order())
    A = (np.random.random(N,) < 0.5).astype(int)
    A[A == 0] = -1
    return A, N

def initialize_T(params, N, A):
    
    #Assert we passed the correct parameters:
    assert 'psi' in params
    assert 'theta0' in params
    assert 'theta1' in params

    #Compute the reporting rates:
    psi = params['psi']
    if type(psi) == np.ndarray: psi = psi.flatten()

    #Initialize T respecting the PU assumption:
    T = (np.random.random(N,) < psi*A).astype(int)
    
    return T

def compute_psi(params, covariates=None, M=None):
    """
    Computes psi based off betas and covariates
    """
    #Check how many covariates we have:
    if M is None: M = 0 if covariates is None else len(np.array(covariates).T)

    #If no covariates, single psi:
    if covariates is None or M == 0:
        psi = expit(params['alpha0'])
        
    #Otherwise, lineraly combine them with covariates:
    else:
        alphas = [params.get(f'alpha{k}') for k in range(M+1)]
        psi = expit(np.dot(np.insert(covariates, 0, 1, axis=1), alphas))

    return psi

def initialize_params(starting_params=None, covariates=None, M=0):
    """
    Provides and verifies inegrity of initial
      guess for the parameters of the MCMC
    
    Parameters
    ----------
    starting_params : None or dict, default None
        dictionary of starting parameter values. if
        None, initial parameters are random

    M : int, default 0
       number of covariates to use in psi regression
        
    Returns
    ----------
    initial_params: dict
        dictionary with keys `alpha0`... `alphaM`, `theta0`, and `theta1`
    """

    #In case starting params are not None, copy:
    if starting_params is not None:
        params = starting_params.copy()
        
    #Otherwise, random initialization:
    else:
        alpha_dict = {f'alpha{k}': np.random.random() for k in range(M+1)}
        theta0 = -np.exp(np.random.normal(scale=0.5)) #Guess Pr(A=1) < Pr(A=0)
        theta1 = np.exp(np.random.normal(scale=0.5))  #Guess correlation > 0
        params = alpha_dict|{'theta0':theta0, 'theta1':theta1}
        
    #Verify integrity of parameters:
    for k in range(M+1): assert f'alpha{k}' in params
    assert 'theta0' in params
    assert 'theta1' in params

    #Include psi:
    if 'psi' not in params: params['psi'] = compute_psi(params, covariates, M)
    
    return params

############################################################################
# PARAMETER CONFIGURATIONS

def add_default_param_HP(user_hyperparams=None):

    #Default values for sampling psi/alphas:
    psi_HP = {'psi_method': 'Reg',
              'psi_Beta_priormean': 0.5,
              'psi_Beta_priorstrength': 2.,
              'psi_Reg_warmup': 50,
              'psi_Reg_model': None,
              'psi_Reg_priormean': None,
              'psi_Reg_priorsd': None}

    #Default values for sampling theta:
    theta_HP = {'theta_method':'SVE',
                'theta_SVE_warmup': 100,
                'theta_SVE_auxiliary': 50,
                'theta_SVE_proposalsigma': 0.2,
                'theta_SVE_adaptive': True,
                'theta_SVE_adaptiveiter': 50,
                'theta_SVE_adaptivebounds':(0.25, 0.6),
                'theta_SVE_adaptiverate': 0.15,
                'theta_SVE_start':[0., 0.3],
                'theta_SVE_priormean':[0., 0.3],
                'theta_SVE_priorsigma': [0.2, 0.1],
                'theta_Exact_theta0bounds': (-2, 2),
                'theta_Exact_theta1bounds': (-2, 2),
                'theta_Exact_nthetas': 200}

    #Default values:
    HP = psi_HP|theta_HP
    
    #Update default values with user hyperparameters:
    if user_hyperparams is not None:
        HP.update(user_hyperparams)
    
    return HP

############################################################################
#SAMPLING THE PARAMETERS

def sample_params(A, T, params,
                  graph,
                  covariates=None,
                  HP=add_default_param_HP(),
                  debug=False):
    """
    Samples alpha0...alphaM, theta0, and theta1 given A and T
    
    Params
    ----------
    A : np.array
      each entry is -1 or 1 corresponding to a node,
      depending of whether incident occured
    T : np.array
      each entry is 0 or 1 corresponding to a node,
      depending of whether incident was reported
    params : dict
      current value of parameters. Usually ignored,
      but used by some methods for warm starts
    graph : Networkx.Graph
      graph object whose nodes correspond to spatial
      locations, must be labeled 0...N-1
    covariates : np.Array of shape N x M or None
        demographic covariates per node that will be used
        in the regression to determine reporting rate. If None,
        report rate is fixed across all nodes
    
    HP : dict
      custom hyperparameters for the sampling
    debug : list or Bool, default False
    
    Returns
    ----------
    sampled_params : dict
    """
    
    #Adding current parameter values for hyperparameters:  
    HP['theta_SVE_start'] = [params['theta0'], params['theta1']]
    M = 0 if covariates is None else len(np.array(covariates).T)
    HP['psi_Reg_start'] = [params[f'alpha{k}'] for k in range(M+1)]
        
    #Sample alphas:
    if M == 0: HP['psi_method'] = 'Beta'
    alpha = sample_alpha(A, T, covariates, HP)
    alpha_dict = {f'alpha{k}': alpha_k for k,alpha_k in enumerate(alpha)}
    if M == 0: alpha_dict['psi'] = expit(alpha_dict['alpha0'])
    
    #Sample thetas:
    theta, debug_args = sample_theta(A, graph, HP, debug)
    theta_dict = {f'theta{k}': theta_k for k,theta_k in enumerate(theta)}
    
    #Create the dictionary:
    params = alpha_dict|theta_dict
                      
    #Include psi:
    if 'psi' not in params: params['psi'] = compute_psi(params, covariates, M)
    
    return params, debug_args

##########################
# Sampling psi and alpha:

def sample_alpha(A, T,
                 covariates=None,
                 HP=add_default_param_HP()):
    """
    Samples alpha0...alphaM given A and T
    
    Params
    ----------
    A : np.array
      each entry is -1 or 1 corresponding to a node,
      depending of whether incident occured
    T : np.array
      each entry is 0 or 1 corresponding to a node,
      depending of whether incident was reported
    covariates : np.Array of shape N x M or None
        demographic covariates per node that will be used
        in the regression to determine reporting rate. If None,
        report rate is fixed across all nodes
    
    HP : dict
      custom hyperparameters for the sampling
    
    Returns
    ----------
    alphas : np.array of length M
    debug_args : list
    """
                     
    #Find out if our method is valid:
    method = HP['psi_method'].lower()
    assert method in ['beta', 'reg']

    #Read hyperparamenters for Beta sampling:
    if method == 'beta':
        alphas = beta_sample_alpha(A, T,
                                   priormean=HP['psi_Beta_priormean'],
                                   priorstrength=HP['psi_Beta_priorstrength'])
        
    #Read hyperparameters for regression sampling:
    elif method == 'reg':

        #We will return the model to update and avoid rebuild:
        alphas, model = reg_sample_alphas(A, T, covariates,
                                          burnin=HP['psi_Reg_warmup'],
                                          prior_means=HP['psi_Reg_priormean'],
                                          prior_sds=HP['psi_Reg_priorsd'],
                                          model=HP['psi_Reg_model'],
                                          return_model=True,
                                          initialization=None)
        HP['psi_Reg_model'] = model

    return alphas

def beta_sample_alpha(A, T, priormean=0.5, priorstrength=2):
    """
    Samples psi given A and T using either the MLE estimate or
      Bayesian inference with a Beta prior. That is, there
      are no covariates! We sample the logit of the report
      rate for convenience (alpha_0)
    
    Params
    ----------
    A : np.array
      each entry is -1 or 1 corresponding to a node, depending
      of whether incident occured
    T : np.array
      each entry is 0 or 1 corresponding to a node, depending
      of whether incident was reported
      
    priormean : float in (0, 1), default 0.5
      if not drawing MLE, mean of the beta prior on psi
    priorstrength : float in (0, inf), default 2
      if not drawing MLE, strength of the beta prior on psi
    
    A and T must satisfy the PU assumption: if A=-1 then
      the corresponding T=0.
    
    In the Beta prior, the mean corresponds to the
      believed proportion of positive reports and the
      strength is akin to the sample size (the larger,
      the more confident we are in our belief)
    
    Returns
    ----------
    alpha0 : float
    """
    #Assert the PU assumption:
    assert 0 not in A + T
    
    #Select the reports for nodes with A=1:
    reports_A1 = T[A == 1]

    #Assert the Beta prior parameters are good:
    assert priormean>=0 and priormean<=1
    assert priorstrength>0
    
    #Find number of reports and failed reports:
    positive_rep = (reports_A1 == 1).sum()
    negative_rep = (reports_A1 == 0).sum()
    
    #This gives the two params of the posterior:
    beta_alp = positive_rep + priorstrength*priormean
    beta_bet = negative_rep + priorstrength*(1-priormean)
    
    sampled_psi = np.random.beta(beta_alp, beta_bet)

    return [logit(sampled_psi)]

##########################
# Sampling theta:
def sample_theta(A, graph, 
                 HP=add_default_param_HP(),
                 debug=False):
    """
    Samples theta0 and theta1
    
    Params
    ----------
    A : np.array
      each entry is -1 or 1 corresponding to a node,
      depending of whether incident occured
    graph : Networkx.Graph
      graph object whose nodes correspond to spatial
      locations, must be labeled 0...N-1
    HP : dict
      custom hyperparameters for the sampling
    debug : Bool or list
    
    Returns
    ----------
    theta0, theta1 : tuple of floats
    debug_args : dict
    """
                     
    #Find out if our method is valid:
    method = HP['theta_method'].lower()
    assert method in ['exact', 'sve']

    #Read hyperparamenters for exact sampling:
    if method == 'exact':
        raise NotImplementedError('Please use SVE instead')
        
    #Read hyperparameters for SVE sampling:
    elif method == 'sve':

        #Return SVE output and acceptance rate:
        SVE_out, acc = sample_SVE(graph,
                                  A,
                                  n_iterations=HP['theta_SVE_warmup'],
                                  theta_start=HP['theta_SVE_start'],
                                  proposal_sigma=HP['theta_SVE_proposalsigma'],
                                  prior_mean=HP['theta_SVE_priormean'],
                                  prior_sigma=HP['theta_SVE_priorsigma'],
                                  n_iterations_aux=HP['theta_SVE_auxiliary'],
                                  debug=debug,
                                  return_acceptance_rate=True)
        
        #Be careful here as we may be returning the debug stats!
        sampled_theta0, sampled_theta1 = SVE_out[-1,:2] if debug else SVE_out[-1]
        debug_args = dict() #NO DEBUG FOR NOW
        debug_args['theta_SVE_acceptance'] = acc  
                     
    return [sampled_theta0, sampled_theta1], debug_args
                     
############################################################################
#SAMPLING THE VARIABLES

def update_AT(A, T, params, graph, fixed_T=False):
    
    #Assert arrays have expected shape:
    N = len(A)
    assert len(T)==N
    assert graph.order()==N
    
    #Randomize the order we update the variables:
    order = list(range(2 * N))
    np.random.shuffle(order)

    #Go trough the 2N variables one at a time:
    for variable_idx in order:
        
        #In case we have variable index 0,1,...N-1 we update A:
        if variable_idx < N:
            
            #Find the neighbors of the node, and get their A values:
            neighbor_idx = list(graph.neighbors(variable_idx))
            psi = params['psi'][variable_idx] if np.size(params['psi']) != 1 else params['psi']
            A[variable_idx] = sample_A(A[neighbor_idx], T[variable_idx], params, psi)
            
        #In case the index is above N, we update T:
        else:
            #If we observe T, then we pass this step:
            if not fixed_T:
                T_idx = variable_idx - N
                psi = params['psi'][T_idx] if np.size(params['psi']) != 1 else params['psi']
                T[T_idx] = sample_T(A[T_idx], psi)
            
    return A, T

def sample_A(A_neighbors, T, params, psi):
    
    #Given T=1, due to PU assumption, A=1
    if T == 1:
        sampled_A = 1
    
    #Given T=0:
    else:
        tau = params['theta0'] + params['theta1']*A_neighbors.sum()
        p = expit(2*tau)            # Pr(A=1 | N(A), theta)
        p_A1 = (1-psi)*p            # Pr(A=1 | N(A), theta, T=0)
        p_A0 = 1 - p                # Pr(A=0 | N(A), theta, T=0)

        #Assertions:
        assert p>=0 and p<=1
        assert p_A1>=0 and p_A1<=1
        assert p_A0>=0 and p_A0<=1
        
        sampled_A = 1 if np.random.random() < (p_A1/(p_A1 + p_A0)) else -1
    
    return int(sampled_A)

def sample_T(A, psi):
    
    p = psi*(A + 1)/2
    assert p>=0 and p<=1
    sampled_T = np.random.binomial(n=1, p=p)
    
    return int(sampled_T)