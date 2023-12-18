############################################################################
# Functions that include baseline models (spatial correlation and GP)
############################################################################

import numpy as np
import networkx as nx

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

############################################################################

def get_trivial_probability(T, graph, fix_reported_nodes=True):
    """
    Get an array of trivial report/incident probabilities
        from a set of existing reports and a network structure. The
        trivial probabilities are computed as, for node i, the fraction
        of neighbors of i with a report. That is:

        P(A_i) = #(nbrs of i with a report)/#(nbrs of i)

    In case the node itself receives a report in the training
        period, we can choose to fix Pr(A_i)=1 similar to 
        the Bayesian model's behavior.

    We use this metric to represent P(A_i) and P(T_i) with no
        distinction---as there is no concept of under-reporting
        in the trivial model.

    Parameters
    ----------
    T : np.Array
        reports in the training period (0 or 1)
        
    graph : Networkx.Graph
        graph object whose nodes correspond to spatial
        locations, must be labeled 0...N-1
        
    fix_reported_nodes : Bool
        when True, assign P(A_i)=1 to all nodes with 
        a report T_i=1

    Returns
    ----------
    probabilities: np.Array
        inferred probabilities
    """

    #Get the adjacency matrix:
    A = nx.adjacency_matrix(graph)

    #Compute the neighbor fraction and degree of each node:
    neighbors_with_report = A.dot(T)
    degrees = A.sum(axis=1)

    #Compute the probability:
    probabilities = neighbors_with_report/degrees
    if fix_reported_nodes: probabilities[T==1] = 1

    return probabilities

def get_GP_probability(T, gdf, scale_by=1000, loc=1):
    """
    Get an array of report/incident probabilities from a set of
        existing reports and spatial locations with a Gaussian
        Process. The GP is trained on nodes with a report. We
        use an RBF kernel on the scaled centroid locations.

    We use this metric to represent P(A_i) and P(T_i) with no
        distinction---as there is no concept of under-reporting
        in the GP model.

    Parameters
    ----------
    T : np.Array
        reports in the training period (0 or 1)
        
    gdf : geopandas.GeoDataFrame
        geodataframe (in projected geometry) with
        node locations
        
    scale_by : float
        numerator to divide centroid positions by,
        typically on the order of magnitude of
        the coordinates

    loc : float
        standard deviation of the RBF kernel

    Returns
    ----------
    probabilities: np.Array
        inferred probabilities
    """

    #Get the array of locations and scale:
    X = np.array([gdf.centroid.x.values, gdf.centroid.y.values]).T
    X_scale = (X - np.mean(X, axis=0))/scale_by

    #Define the Gaussian process
    kernel = RBF(loc)
    GP = GaussianProcessRegressor(kernel=kernel, normalize_y=False, optimizer=None)

    #Fit to nodes with a report:
    X_train = X_scale[T == 1]
    GP_fitted = GP.fit(X_train, [1]*len(X_train))

    #Predict:
    probabilities = np.clip(GP_fitted.predict(X_scale), 0, 1)

    return probabilities