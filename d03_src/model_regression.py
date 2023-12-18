import numpy as np
import pandas as pd
import bambi as bmb
import pymc.math as pmm
import pymc as pm

############################################################################

def reg_sample_alphas(A, T, covariates=None, burnin=100,
                      prior_means=None, prior_sds=None,
                      model=None, return_model=False,
                      initialization=None):
    """
    Sample alphas (coefficients) using a logistic regression
      of ground truths and reports. Coefficients are
      given normal priors.

    Params
    ----------
    A : np.array
      each entry is -1 or 1 corresponding to a node, depending
        of whether incident occured
      
    T : np.array
      each entry is 0 or 1 corresponding to a node, depending
        of whether incident was reported
      
    covariates : np.array N x M or None
      array with covariates (demographic features) for
        each node. Must have N rows. If None, only intercept
        is fit in the regression (all nodes have the same
        psi)
    
    burnin : int, default 100
       number of burn-in samples in the bayesian sampling
       
    prior_means : array of length M+1 or None
       mean of each coefficient for the prior distribution
       
    prior_sds : array of length M+1 or None
       sd of each coefficient for the prior distribution
       
    model : pymc.Model or None
       logistic model with MutableData, if not passed will
       be built from scratch

    return_model : Bool
       whether to return the built model, which can then be
       updated in subsequent iterations

    initialization : dict or None, default None
       values to initialize the sampler. If dict, keys must be 'Intercept' and
       each of the 'labels' variables. If list, length must be M+1.
    
    A and T must satisfy the PU assumption: if A=-1 then
      the corresponding T=0.
    
    Returns
    ----------
    alphas : np.array of length M+1
    
    """
    #Assert the PU assumption:
    assert 0 not in A + T
    
    #We can only use for the regression the
    # locations where an incident occured:
    positive_nodes = np.nonzero(A+1)[0]

    #We select the response variable and the
    # covariates for these locations:
    y = T[positive_nodes]
    X = np.insert(covariates[positive_nodes], 0, 1, axis=1)
    M = X.shape[1]

    #If we did not pass priors, use standard
    # uninformative priors:
    if prior_means is None: prior_means = [0.]*M
    if prior_sds is None: prior_sds = [2.]*M

    #Build the model:
    if model is None: model = pymc_logistic_model(y, X, prior_means, prior_sds)

    #Check initial values:
    if type(initialization) == list: initialization = dict([f'alpha{k}' for k in range(len(initialization))], initialization)
    #Sample:
    with model:
        pm.set_data({'T':y, 'X':X})
        samples = pm.sample(draws=1, tune=burnin, chains=1,
                            initvals=initialization, init='adapt_diag',
                            progressbar=False, compute_convergence_checks=False)
    
    alphas = samples.posterior['alpha'].values.flatten()
    
    return alphas if not return_model else (alphas, model)

############################################################################

def pymc_logistic_model(y, X, prior_means, prior_sds):
    """
    Build a logistic model with Pymc
    """
    
    with pm.Model() as logistic_model:
        #Define the observable data:
        covariates = pm.MutableData('X', X)
        reports = pm.MutableData('T', y)
        #Define the priors on our coefficients:
        alpha = pm.Normal('alpha', mu=prior_means, sigma=prior_sds, shape=len(prior_means))
        #Specify the distribution of T:
        pm.Bernoulli('obs', logit_p=pmm.matrix_dot(covariates, alpha), observed=reports)
    return logistic_model

############################################################################

def sample_alphas_pymc3(y, X, prior_means, prior_sds, burnin=100, labels=None, return_inference_data=False):

    """
    This uses pymc3 which is deprecated
    """
    
    #Verify the input:
    assert len(prior_means) == len(prior_sds)
    assert X.shape[0] == len(y)
    assert X.shape[1]+1 == len(prior_means)
    
    #Name the covariates:
    if labels is None or len(labels) != len(X.shape[1]+1):
        labels = [f'X{k+1}' for k in range(X.shape[1])]
    
    #Specify the model:
    with pm.Model() as model:
          
        # Priors
        intercept_prior = {'Intercept': pm.Normal.dist(mu=prior_means[0], sigma=prior_sds[0])}
        regressor_prior = {var: pm.Normal.dist(mu=prior_means[k+1], sigma=prior_sds[k+1]) for k, var in enumerate(labels)}
        prior = intercept_prior|regressor_prior
        
        # Model
        pm.glm.GLM(y=y, x=X, labels=labels, intercept=True,
                   priors=prior,
                   family=pm.glm.families.Binomial())
        idata = pm.sample(1, chains=1, tune=burnin, return_inferencedata=False)
        
    alphas = np.array([idata[alpha].flatten() for alpha in ['Intercept']+labels])
    
    return alphas if not return_inference_data else (alphas, idata)

############################################################################

def sample_alphas_bambi(y, X, prior_means, prior_sds,
                        initialization=None,
                        burnin=100,
                        labels=None, return_inference_data=False):

    #Verify the input:
    N, M = X.shape[0], X.shape[1]

    assert len(y) == N                       
    assert len(prior_means) == M+1
    assert len(prior_sds) == M+1
    assert len(labels) == M+1

    #Create a dataframe:
    data = pd.DataFrame(X, columns=labels[1:])
    data['T'] = y
    
    # Priors
    priors = {var: bmb.Prior('Normal', mu=prior_means[k], sigma=prior_sds[k])
              for k, var in enumerate(labels)}
    
    #Specify the model:
    formula = f"T ~ 1 + {' + '.join(labels[1:])}"
    model = bmb.Model(formula, data=data, family="bernoulli", priors=priors)
    
    #Sample:                     
    idata = model.fit(draws=1, tune=burnin, chains=1,
                      initvals=initialization,
                      progressbar=False, compute_convergence_checks=False)
    alphas = np.array([idata.posterior[alpha].values.flatten()[-1] for alpha in labels])

    return alphas if not return_inference_data else (alphas, idata)