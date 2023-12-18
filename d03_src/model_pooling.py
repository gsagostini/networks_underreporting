##########################################################################################################
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

##########################################################################################################
#This file contain functions to pool posteriors for the same parameter using a Bayes factor.

##########################################################################################################
# 1. BAYES FACTOR FUNCTIONS
    
def get_combined_posterior_from_samples(samples_list,
                                        prior_mean=0., prior_std=1.,
                                        estimate_type='mean', CI_alpha=0.95, return_pdf=False,
                                        plot=False):
    """
    Given an array of samples from Pr(alpha|T_k) for multiple
        observations of T, compute the combined posterior 
        distribution Pr(alpha|{T_k}_k). The parameter alpha is
        assumed to have a normal prior distribution.

    Returns by default a parameter estimate (mean or median) and
        the desired CI. If asked, also returns the combined pdf.
    
    Parameters
    ----------
    samples_list : list of lists
        each entry is a list of samples from one 
        particular posterior. note that not all entries
        may have the same length: due to the thinning fraction
        in the MCMC model, posteriors may have different
        number of elements

    prior_mean, prior_std : floats
        parameters of the Normal prior distributions

    estimate_type : `mean` or `median`
        what parameter estimate to return

    CI_alpha : float
        what confidence interval to return

    return_pdf : Bool
        whether to return the pdf posterior

    plot : Bool
        whether to visualize the posteriors and the
        combined distribution
        
    Returns
    ----------
    estimate
        point estimate for parameter alpha

    CI
        confidence interval requested from CI_alpha

    pdf (optional)
        combined posterior pdf
    """

    #Build the prior pdf:
    prior = lambda x: stats.norm.pdf(x, loc=prior_mean, scale=prior_std)
                                            
    #Build the posterior pdfs:
    posteriors = [fit_normal_pdf_to_samples(samples) for samples in samples_list]
    
    #Combine the posterior:
    combined_posterior = combine_posteriors(posteriors, prior)

    #Get the estimate and CI:
    percentiles_to_attain = [(1-CI_alpha)/2, (1+CI_alpha)/2]
    CI, estimate = summarize_pdf(combined_posterior,
                                 pctiles=percentiles_to_attain,
                                 estimate_type=estimate_type)
    #Plot the functions:
    if plot:
        fig, ax = plt.subplots(figsize=(9, 5))
        #Get the pdf support:
        x = np.linspace(-5,5,1000)
        #Plot the individual posteriors:
        for idx, posterior in enumerate(posteriors):
            _ = ax.plot(x, get_normalized_pdf_at_points(posterior, x),
                        label = f'posterior #{idx+1}')
        #Plot the combined posterior:
        _ = ax.plot(x, get_normalized_pdf_at_points(combined_posterior, x), label = f'combined posterior')
        _ = ax.legend()
        plt.show()

    if return_pdf:
        return estimate, CI, combined_posterior
    else:
        return estimate, CI
    
def combine_posteriors(posteriors, prior):
    """
    Given an array of posterior pdfs and a prior pdf,
        combine the posteriors into a single pdf
        using the Bayes factor described in appendix A
        of the paper.
    
    Parameters
    ----------
    posteriors : np.Array
        array of posterior pdfs of the parameter

    prior : function
        prior pdf of the parameter
        
    Returns
    ----------
    function
        unnormalized combined posterior pdf
    """
    
    # Compute the Bayes factors on log scale:
    log_bayes_factor = lambda x: [np.log(posterior(x)) - np.log(prior(x)) for posterior in posteriors]

    #Multiply all these terms:
    bayes_factor_prod = lambda x: np.sum(log_bayes_factor(x))
    
    # Multiply by the last prior term and exponentiate:
    combined_posterior = lambda x: np.exp(bayes_factor_prod(x) + np.log(prior(x)))
    
    return combined_posterior
    
##########################################################################################################
# 2. Functions to process pdfs:

def fit_normal_pdf_to_samples(samples):
    """
    Fits a normal distribution to a ser
      of draws from it
    
    Parameters
    ----------
    samples : np.Array
        
    Returns
    ----------
    function
        pdf of the distribution
    """
    
    #Estimate the parameters from the sample:
    mean,std = np.mean(samples), np.std(samples)
    
    #Get normal pdf from mean and std:
    normal_pdf = lambda x: stats.norm.pdf(x, loc=mean, scale=std)
    
    return normal_pdf

def get_normalized_pdf_at_points(pdf, points):
    """
    Given a (potentially un-normalized) pdf and
        an array of points, obtain the value of
        the pdf at each point so that the total
        values sum to 1.
    
    Parameters
    ----------
    pdf : function
        one-argument pdf function

    points : np.Array
        points to evaluate the pdf
        
    Returns
    ----------
    np.Array       
    """
    #Obtain the pdf value at every point:
    pdf_list = [pdf(x) for x in points]
    
    #Compute the normalization constant and normalize:
    C = np.sum(pdf_list)
    normalized_pdf_list = [p/C for p in pdf_list]
    
    return normalized_pdf_list

def summarize_pdf(pdf,
                  xmin=-10, xmax=10, n_points=10_000,
                  pctiles=[0.025, 0.975],
                  estimate_type='mean'):
    """
    Given a (potentially un-normalized) pdf and
        an array of points, obtain the value of
        percentiles and a point estimate.
    
    Parameters
    ----------
    pdf : function
        one-argument pdf function

    xmin, xmax : floats
        bounds of the pdf support

    n_points
        bounds of the pdf support

    pctiles : np.Array
        what percentiles to compute

    estimate_type : `mean` or `median`
        what point estimate to return
        
    Returns
    ----------
    np.Array
        percentiles for the distribution
    float
        estimate
    """
    #Create an array of possible values:
    points = np.linspace(xmin, xmax, n_points)

    #Get the pdf at each of those points:
    pdf_at_points = get_normalized_pdf_at_points(pdf, points)

    #Compute the cdf:
    cdf = np.cumsum(pdf_at_points)
    cdf = cdf/np.max(cdf)

    #Interpolate to find the percentiles:
    percentiles = np.interp(pctiles, cdf, points)

    #Find the estimate:
    if estimate_type == 'median': estimate = np.interp([0.5], cdf, points)[0]
    if estimate_type == 'mean': estimate = np.average(points, weights=pdf_at_points)
    
    return percentiles, estimate

##########################################################################################################