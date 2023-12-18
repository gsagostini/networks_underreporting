import numpy as np
import networkx as nx
from scipy.stats import norm
from scipy.special import expit

#from tqdm import tqdm
#####################################################################
# SINGLE-VARIABLE EXCHANGE FOR OBSERVED MODEL (A, no T)

def sample_SVE(graph,
               A,
               n_iterations=1000,
               theta_start=[0., 0.],
               proposal_sigma=0.2,
               prior_mean=[0., 0.3],
               prior_sigma=[0.2, 0.1],
               n_iterations_aux=50,
               use_Gibbs_aux=False,
               debug=False,
               return_acceptance_rate=False):
    """
    Sample the parameters theta using the Single-Variable
      Exchange Algorithm (SVAE) proposed by Murray et al. in
      https://arxiv.org/ftp/arxiv/papers/1206/1206.6848.pdf
    
    Parameters
    ----------
    graph : Networkx.Graph
        graph object whose nodes correspond to spatial
        locations, must be labeled 0...N-1
    A : np.array
        true values for the hidden state of each node,
        either +1 or -1 (both cases must appear)
    n_iterations : int, default 5000
        total number of iterations in the chain
    theta_start: list
        list of starting theta values. default is
        two parameters starting at 0.
    proposal_sigma : float
        standard deviation for proposal step
    n_iterations_aux : int
        total number of iterations in auxiliary
        variable generation
    debug : Boolean
        whether to return not only theta but also
        the acceptance probabilities and the auxiliary
        variables
    return_acceptance_rate : Boolean
        whether to return the acceptance rate (the
        number of accepted proposals divided by the
        number of iterations)
        
    Returns
    ----------
    sampled_theta: list
    """
    
    #Initalization of parameters:
    N = graph.order()
    adj_matrix = nx.adjacency_matrix(graph)
    d = len(theta_start)
    
    #Check that the passed A is coherent:
    assert len(A) == N
    assert np.all(abs(A) == 1)
    
    #Create an array for thetas, priors, and densities:
    theta = np.zeros((n_iterations+1, d))
    priors = np.zeros(n_iterations+1)
    densities = np.zeros(n_iterations+1)
    acceptance_probabilities = np.zeros((n_iterations+1, 7))
    w_means = np.zeros(n_iterations+1)
    accepted_proposals = 0
    
    #For the density, we will constantly need the values of V0
    # and V1 that depend only on (fixed) A and the graph:
    V_A = get_V(A, adj_matrix)
    
    #Initial values
    theta[0] = theta_start
    priors[0] = get_prior(theta[0], prior_mean, prior_sigma, log=True)
    densities[0] = get_density(theta[0], V=V_A, log=True)
    acceptance_probabilities[0] = [1. for k in range(7)]
    w_means[0] = np.mean(A)
    
    #Iterate:
    for iteration in range(n_iterations):
        
        #Propose theta_prime:
        theta_prime = theta[iteration] + np.random.randn(d)*proposal_sigma
        prior_prime = get_prior(theta_prime, prior_mean, prior_sigma, log=True)
        density_prime = get_density(theta_prime, V=V_A, log=True)
        
        #Generate the auxiliary variable W:
        if iteration==0:
            starting_w = A.copy()
        else:
            starting_w = w.copy()
        w = sample_from_ising(graph, theta_prime, starting_w,
                              n_iterations_aux, use_Gibbs_aux)
        w_means[iteration+1] = np.mean(w)
        
        V_w = get_V(w, adj_matrix)
        density_w_prime = get_density(theta_prime, V=V_w, log=True)
        density_w = get_density(theta[iteration], V=V_w, log=True)
        
        #Compute the acceptance ratio (in log space)
        num = prior_prime       + density_prime        + density_w
        den = priors[iteration] + densities[iteration] + density_w_prime
        a = np.exp(num - den)
        
        acceptance_probabilities[iteration+1] = [a,
                                                 prior_prime, density_prime, density_w_prime,
                                                 priors[iteration], densities[iteration], density_w]
        
        #Accept or maintain:
        r = np.random.uniform()
        if r < a:
            theta[iteration+1]     = theta_prime
            priors[iteration+1]    = prior_prime
            densities[iteration+1] = density_prime
            accepted_proposals += 1
        else:
            theta[iteration+1]     = theta[iteration]
            priors[iteration+1]    = priors[iteration]
            densities[iteration+1] = densities[iteration]
    
    #Compute the acceptance rate:
    acceptance_rate = accepted_proposals/n_iterations
    
    #Decide what is the main return user wants:
    return_arr = np.hstack([theta, acceptance_probabilities, w_means.T]) if debug else theta
    return (return_arr, acceptance_rate) if return_acceptance_rate else return_arr

############################################################################
# GENERATE AUXILIARY VARIABLE (Gibbs):

def sample_from_ising(graph, theta,
                      starting_A=None,
                      n_iterations=500,
                      use_Gibbs=False):

    """
    Sample the spins of an Ising model, i.e. samples
      a binary vector (+-1) given a graph structure
      with probability proportional to
    
    q(A|theta) = exp{theta_0*V_0(A) + theta_1*V_1(A)}
    
        V_0(A) = sum(A_i)             # mean incidence
        V_1(A) = sum(A_i*A_j*E_ij)    # correlations
    
    Parameters
    ----------
    graph : Networkx.Graph
        graph object whose nodes correspond to spatial
        locations, must be labeled 0...N-1
    theta : np.array, list, or tuple of lenght 2
        true values for two parameters
    starting_A : np.array
        initial configuration for the spins. If None, 
        begin with A_i ~ Bern(theta_0)
    n_iterations : int, default 500
        total number of iterations in the chain
    use_Gibbs : Bool, default False
        Gibbs sampling is slower to converge but
        guaranteed to work with negative attractions
        (theta_1 < 0)---in which case the function will
        default to use Gibbs. If use_Gibbs is True, the
        function will always use this method; if False, 
        it will prefer the Swendsen-Wang algorithm
        
    Returns
    ----------
    sampled_A: list
    """
    
    #Determine the starting A:  
    if starting_A is None:
        p1 = expit(-2*theta[0])
        starting_A = (np.random.random(N,) < p1).astype(int)
    starting_A[starting_A == 0] = -1
    
    #Determine the method to be used:
    if use_Gibbs or theta[1] <= 0:
        sampled_A = sample_ising_Gibbs(graph, theta, starting_A, n_iterations)
    else:
        sampled_A = sample_ising_SwendsenWang(graph, theta, starting_A, n_iterations)
        
    return sampled_A

def sample_ising_Gibbs(graph, theta,
                       starting_A,
                       n_iterations=100,
                       return_all=False):

    #Get the parameters:
    A = np.array(starting_A.copy())
    N = len(A)
    assert graph.order() == N
    if return_all: all_A = np.zeros((n_iterations, len(A)))
    
    #Run the chain:
    for iteration in range(n_iterations):
        
        #Random order to update the nodes:
        order = list(range(N))
        np.random.shuffle(order)
        
        for node in order:
            
            #Find the neighbors of the node, and get their A values:
            ngb_iter = graph.neighbors(node)
            ngb_A = A[list(ngb_iter)]
            
            #Update the variable A with sampling:
            A[node] = sample_single_A(ngb_A.sum(), theta)
            
        if return_all: all_A[iteration] = A
            
    return A if not return_all else all_A

def sample_single_A(A_neighbors_sum, theta):
    
    tau = theta[0] + theta[1]*A_neighbors_sum
    p_A1 = np.exp(tau)
    p_A0 = np.exp(-tau)
    p = p_A1/(p_A1 + p_A0)
    sampled_A = 1 if np.random.random() < p else -1
    
    return int(sampled_A)
    
def sample_ising_SwendsenWang(graph, theta,
                              starting_A,
                              n_iterations=100,
                              return_all=False):
    
    #Ensure we have an attractive potential:
    assert theta[1] >= 0
    
    #Get the parameters:
    A = np.array(starting_A.copy())
    p_delete = np.exp(-2*theta[1])
    N = len(A)
    assert graph.order() == N
    if return_all: all_A = np.zeros((n_iterations, len(A)))
    
    #Run the chain:
    for iteration in range(n_iterations):
        
        #Find the monochromatic edges:
        M = np.array([edge for edge in graph.edges() if A[edge[0]] == A[edge[1]]])

        #Subsample the monochromatic edges:
        M_prime = [edge for edge in M if np.random.random() > p_delete]

        #Create the SW graph
        SW_graph = nx.Graph()
        SW_graph.add_nodes_from(range(N))
        SW_graph.add_edges_from(M_prime)
        
        #Flip the signals of components:
        for component in nx.connected_components(SW_graph):
            A[list(component)] = 1 if np.random.random() < expit(2*theta[0]*len(component)) else -1
        
        if return_all: all_A[iteration] = A
    
    return A if not return_all else all_A
    
############################################################################
# DENSITIES:

def get_prior(x_arr, mu_arr=[0., 0.], sigma_arr=[1., 1.], log=True):
    
    if log:
        pi = 0
        for theta, mean, std in zip(x_arr, mu_arr, sigma_arr):
            pi += norm.logpdf(theta, loc=mean, scale=std)
    
    else:
        pi = 1
        for theta, mean, std in zip(x_arr, mu_arr, sigma_arr):
            pi *= norm.pdf(theta, loc=mean, scale=std)
    return pi

def get_V(assignment, graph_matrix):
    V_0 = sum(assignment)
    V_1 = 0.5*np.dot(assignment, graph_matrix.dot(assignment))
    return np.array([V_0, V_1])

def get_density(theta, graph_matrix=None, A=None, V=None, log=True):
    #un-normalized!
    if V is None:
        V = get_V(A, graph_matrix)
    
    exponent = theta[0]*V[0] + theta[1]*V[1]
    
    if log:
        return exponent
    else:
        return np.exp(exponent)

############################################################################
# ADAPTIVE STEPSIZE:
def adapt_SVE_stepsize(HP,
                       acceptance_list,
                       iteration):

    #Check our hyperparameters contain all keys:
    assert 'theta_SVE_proposalsigma' in HP
    assert 'theta_SVE_adaptive' in HP
    assert 'theta_SVE_adaptivebounds' in HP
    assert 'theta_SVE_adaptiverate' in HP
    assert 'theta_SVE_adaptiveiter' in HP
    
    #Check the routine is warranted:
    if HP['theta_SVE_adaptive']:
        
        #Check we're on an iteration to update:
        if iteration%HP['theta_SVE_adaptiveiter'] == 0 and iteration>0:
            
            #Check what proportion of proposals were accepted and
            # clear the acceptance list:
            acceptance_rate = np.mean(acceptance_list)
            acceptance_list = []
            #Get the new value if the acceptance rate is outside
            # of the desired bounds:
            if acceptance_rate <= HP['theta_SVE_adaptivebounds'][0]:
                multiplier = 1-HP['theta_SVE_adaptiverate']
                HP['theta_SVE_proposalsigma'] *= multiplier
            elif acceptance_rate >= HP['theta_SVE_adaptivebounds'][1]:
                multiplier = 1+HP['theta_SVE_adaptiverate']
                HP['theta_SVE_proposalsigma'] *= multiplier

    return HP, acceptance_list