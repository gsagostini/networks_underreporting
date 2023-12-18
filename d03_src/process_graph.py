#####################################################
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math
from copy import deepcopy

import geopandas as gpd
from scipy.spatial import distance
from libpysal import weights
from pygris import blocks, tracts, block_groups
from pygris.utils import erase_water
from polygon_geohasher.polygon_geohasher import polygon_to_geohashes, geohash_to_polygon

import sys
sys.path.append('../d03_src/')
import process_demographics as prd
import vars

#####################################################
#This file contain functions to process spatial data into networkX graph objects.

#There are 5 classes of functions:

#  1. Graph generators, which build the graph from one of the following input structures:
#      a. Grid graph specifications
#      b. Geohashes
#      c. Census geographies

#  2. Geography trimmers, which remove from the dataset areas that don't satisfy certain requirements such as:
#      a. Areas with low population
#      b. Areas with little land area
#      c. Areas with huge park area (in NYC)

#  3. Graph trimmers, which remove from a graph nodes that don't satisfy certain requirements such as:
#      a. Nodes with too many neighbors
#      b. Edges that are too long (i.e. nodes that are too distant from any other node)

#  4. General graph utilities

#  5. Graph drawing

#####################################################
#GRAPH GENERATORS:

def generate_graph_from_gdf(gdf,
                            node_id_column=None,
                            graph_attributes={'unit':'geohash', 'precision':6},
                            projected_crs=vars._projected_crs,
                            weights_type='rook',
                            enforce_connection=True):
    """
    Takes a geodataframe and turns it into a simple graph
        based on adjacency.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        geographic units that correspond to nodes of the graph. must
        be indexed as 0, ... N-1 for correct results.
    
    node_id_column : str
        column to use in a node_id attribute
    graph_attributes : dict
        dictionary of attribtues for the graph

    weights_type : `rook` or `queen`
        how to connect adjacent regions

    enforce_conenction : bool
        whether to add edges to ensure the graph has a
        a single connected component
        
    Returns
    ----------
    graph : Networkx.Graph
    """
                           
    #Project the gdf:
    projected_gdf = gdf.to_crs(projected_crs)

    #Get a weights object:
    if weights_type.lower() == 'rook':
        weights_object = weights.Rook.from_dataframe(projected_gdf)
    elif weights_type.lower() == 'queen':
        weights_object = weights.Queen.from_dataframe(projected_gdf)

    #Construct the graph:
    graph = weights_object.to_networkx()

    #Set the id attribute:
    if node_id_column is not None:
        _ = nx.set_node_attributes(graph, gdf[node_id_column].to_dict(), name='id')

    #Get the centroids of the nodes (a position dictionary too):
    centroids = projected_gdf.centroid
    positions = {n:(centroid.x, centroid.y) for n, centroid in enumerate(centroids)}

    #Save these values as graph and node attributes:
    _ = nx.set_node_attributes(graph, centroids.x.to_dict(), name='x')
    _ = nx.set_node_attributes(graph, centroids.y.to_dict(), name='y')
    _ = graph.graph.update(graph_attributes|{'positions': positions, 'crs': projected_crs})

    #Enforce connection:
    if enforce_connection:
        #Find the component each node lies in:
        components = [c for c in nx.connected_components(graph)]
        components_dict = {n:find_component(n, components) for n in range(graph.order())}
        _ = nx.set_node_attributes(graph, components_dict, name='component')
        #Add edges:
        graph = make_connected(graph)

    return nx.convert_node_labels_to_integers(graph)

def generate_graph_grid(grid_size):
    """
    Generates a square grid with 
        side length provided i.e. with
        grid_size**2 nodes
    
    Parameters
    ----------
    grid_size : int
        size of the grid
        
    Returns
    ----------
    graph : Networkx.Graph
    """
    grid_raw = nx.grid_2d_graph(grid_size, grid_size, create_using=nx.Graph)
    grid = nx.convert_node_labels_to_integers(grid_raw)
    return grid

def generate_graph_census(census_unit='tracts',
                          state='NY',
                          counties=['New York','Bronx','Kings','Queens','Richmond'],
                          weights_type='rook',
                          remove_zeropop=True,
                          remove_parks=False,
                          remove_water=True,
                          tresh_parkarea=0.75,
                          tresh_population=1,
                          enforce_connection=True,
                          projected_crs=vars._projected_crs):
    """
    Generates a graph from census geographies.
    
    Parameters
    ----------
    census_unit : `tracts` or `blockgroups` or `blocks`
        geographic units to use
    
    state : str
        state the counties are found on
    counties : list
        list of counties to collect geographies from

    weights_type : `Rook` or `Queen`
        how to connect adjacent regions

    remove_water : bool
        whether to remove water areas
        
    remove_zeropop : bool
        wheter to remove areas with low population

    tresh_population : float
        minimum population if remove_zeropop
        
    remove_parks : bool
        whether to remove areas with high park area

    tresh_parkarea : float
        maximum park area allowed if remove_parks
        
    enforce_conenction : bool
        whether to add edges to ensure the graph has a
        a single connected component
        
    Returns
    ----------
    gdf : gpd.GeoDataFrame
    graph : Networkx.Graph
    """
                              
    #Collect the full census geography data:
    if census_unit.lower() in ['tract', 'tracts']:
        census_gdf = tracts(state=state, county=counties)
        
    elif census_unit.lower() in ['blocks', 'block']:
        census_gdf = blocks(state=state, county=counties)
        
    elif census_unit.lower() in ['block groups', 'blockgroups', 'block_groups']:
        census_gdf = block_groups(state=state, county=counties)
        
    else:
        print('Unclear what census unit to use. Using census tracts.')
        census_gdf = tracts(state=state, county=counties)
        
    #Remove water:
    if remove_water:
        census_gdf = erase_water(census_gdf).reset_index(drop=True)
        
    #Remove units with low population:
    if remove_zeropop:
        census_gdf = trim_by_population(census_gdf,
                                        id_col='GEOID',
                                        pop_tresh=tresh_population)
        
    #Remove parks:
    if remove_parks:
        census_gdf = trim_by_parkarea(census_gdf,
                                      id_col='GEOID',
                                      maximum_park_area_fraction=tresh_parkarea)
        
    #Get the graph:
    graph = generate_graph_from_gdf(gdf=census_gdf,
                                    node_id_column='GEOID',
                                    projected_crs=projected_crs,
                                    graph_attributes={'unit':'census', 'precision':census_unit},
                                    weights_type=weights_type,
                                    enforce_connection=enforce_connection)
    
    return census_gdf.to_crs(projected_crs), graph

def generate_graph_geohash(precision=6,
                           boundary_file=vars._nyc_shp,
                           projected_crs=vars._projected_crs,
                           weights_type='rook',
                           remove_water=True,
                           remove_zeropop=True,
                           remove_parks=True,
                           tresh_water=0.5,
                           tresh_parkarea=0.75,
                           tresh_population=1,
                           enforce_connection=True):
    """
    Generates a graph from geohashes.
    
    Parameters
    ----------
    precision : 5, 6, or 7
        geohash precision to use
    
    boundary_file : str
        shapefile containing the city outline

    weights_type : `Rook` or `Queen`
        how to connect adjacent regions

    remove_water : bool
        whether to remove water areas
        
    tresh_water : float
        minimum land area allowed if remove_water
        
    remove_zeropop : bool
        wheter to remove areas with low population

    tresh_population : float
        minimum population if remove_zeropop
        
    remove_parks : bool
        whether to remove areas with high park area

    tresh_parkarea : float
        maximum park area allowed if remove_parks
        
    enforce_connection : bool
        whether to add edges to ensure the graph has a
        a single connected component
        
    Returns
    ----------
    gdf : gpd.GeoDataFrame
    graph : Networkx.Graph
    """
                               
    #Get the boundary:
    polygon_gdf = gpd.read_file(boundary_file)
    boundary_gdf = gpd.GeoDataFrame(geometry=[polygon_gdf.unary_union], crs=polygon_gdf.crs).to_crs('EPSG:4326')

    #Get the geohashes:
    geohashes = polygon_to_geohashes(boundary_gdf.geometry[0], precision, False)
    geohashes_gdf = gpd.GeoDataFrame(geohashes,
                                     geometry=[geohash_to_polygon(geohash) for geohash in geohashes],
                                     columns=['geohash'],
                                     crs='EPSG:4326')
    #Remove water:
    if remove_water:
        geohashes_gdf = trim_by_landarea(geohashes_gdf,
                                         id_col='geohash',
                                         minimum_land_area_fraction=tresh_water)
        geohashes_gdf = gpd.clip(geohashes_gdf.to_crs(projected_crs), boundary_gdf.to_crs(projected_crs))

    #Remove units with low population:
    if remove_zeropop:
        geohashes_gdf = trim_by_population(geohashes_gdf,
                                           id_col='geohash',
                                           pop_tresh=tresh_population)

    #Remove parks:
    if remove_parks:
        geohashes_gdf = trim_by_parkarea(geohashes_gdf,
                                         id_col='geohash',
                                         maximum_park_area_fraction=tresh_parkarea)

    #Get the graph:
    graph_geohash = generate_graph_from_gdf(gdf=geohashes_gdf,
                                            node_id_column='geohash',
                                            projected_crs=projected_crs,
                                            graph_attributes={'unit':'geohash', 'precision':precision},
                                            weights_type=weights_type,
                                            enforce_connection=enforce_connection)

    return geohashes_gdf.to_crs(projected_crs), graph_geohash

#####################################################
#TRIM DATAFRAME:

def trim_by_population(gdf,
                       id_col='GEOID',
                       pop_tresh=1):
    """
    Removes from a geodataframe the polygons whose 
      population is less than a given treshold
    """
    #Include population:
    population = prd.include_covariates(gdf,
                                        col_to_merge_on=id_col,
                                        covariates_names=['population'],
                                        standardize=False)
    #Select the nodes to keep:
    filtered_gdf = gdf.iloc[population >= pop_tresh]
    
    return filtered_gdf.reset_index(drop=True)

def trim_by_parkarea(gdf,
                     id_col='GEOID',
                     maximum_park_area_fraction=0.75):
    """
    Removes from a geodataframe the polygons whose 
      fraction of the area covered by parks is too
      high (above a certain treshold)
    """
    #Include park area:
    park_area = prd.include_park_area(gdf,
                                      id_col=id_col,
                                      include_as_fraction=True)
                         
    #Select the nodes to keep:
    filtered_gdf = gdf.iloc[park_area <= maximum_park_area_fraction]
                         
    return filtered_gdf.reset_index(drop=True)

def trim_by_landarea(gdf,
                      id_col='GEOID',
                      minimum_land_area_fraction=0.75):
    """
    Removes from a geodataframe the polygons whose 
      fraction of the area covered by water is too
      high (land area below a certain treshold)
    """
    #Include land area:
    land_area = prd.include_land_area(gdf,
                                      id_col=id_col,
                                      include_as_fraction=True)
                         
    #Select the nodes to keep:
    filtered_gdf = gdf.iloc[land_area >= minimum_land_area_fraction]
                         
    return filtered_gdf.reset_index(drop=True)

#####################################################
#TRIM GRAPH:

def get_high_degree_nodes(graph, max_degree):
    
    high_degree_nodes = [node for node,degree in dict(graph.degree()).items() if degree > max_degree]
    
    return high_degree_nodes

def get_long_edges(graph, max_edge_length):

    long_edges = []
    for u, v in graph.edges(data=False):
        
        #Collect the location of the nodes:
        u_xy = (graph.nodes[u]['x'], graph.nodes[u]['y'])
        v_xy = (graph.nodes[u]['x'], graph.nodes[u]['y'])
        
        #Compute edge length:
        length = math.dist(u_xy, v_xy)/3.28084 #convert to meters

        #Is the edge long:
        if length > max_edge_length: long_edges.append((u, v))
    
    return long_edges
                              
#####################################################
#GRAPH UTILS:

def augment_graph(graph):
    """
    Takes a graph and adds a dead-end to each node i.e. adds N
      new nodes, each with an edge connecting to an existing
      node. Used to represent the observed and hidden variable
      structure in HMM models.
     
    Nodes are converted to integers with default order. It is
      recommended that input graph has integer-labeled nodes.
      
    Directionality of new edges is always from spatial (old)
      node to new node. Undirected graph is recommended.
    
    Parameters
    ----------
    graph : Networkx.Graph or Networkx.MultiGraph
        graph object whose nodes correspond to spatial
        locations, should be labeled 0...N-1
    
    Returns
    ----------
    augmented_graph : Networkx.Graph or Networkx.MultiGraph
    """
    n = graph.order()
    augmented_graph = nx.convert_node_labels_to_integers(graph)
    augmented_graph.add_edges_from([(k, n+k) for k in range(n)])
    return augmented_graph

def find_component(node, components_list):
    for component_idx, nodes in enumerate(components_list):
        if node in nodes:
            return component_idx
    return None

def make_connected(graph,
                   x_col='x', y_col='y', component_col='component'):
                       
    #Get the nodes with their positions and components on a dataframe:
    nodes = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')
    
    #Find the sizes of the components:
    component_sizes = nodes[component_col].value_counts(sort=True)
    assert len(component_sizes) > 1
    
    #We want to keep just the largest component.
    giant_component = nodes[nodes[component_col]==component_sizes.index[0]]
    giant_component_nodes = giant_component.index.values
    giant_component_xy = giant_component[[x_col, y_col]].values
    
    #One component at a time, let's merge the nodes into the giant component:
    edges_to_add = []
    for component_idx in component_sizes.index[1:]:
        
        #Get the nodes and positions in the component:
        component = nodes[nodes[component_col]==component_idx]
        component_nodes = component.index.values
        component_xy = component[[x_col, y_col]].values
    
        #Find the distances between all points in this component and
        # all points in the giant component:
        pairwise_distances = distance.cdist(component_xy, giant_component_xy)
    
        #Get the index of the minimum distance points:
        min_distance_idx = np.argmin(pairwise_distances)
        u_idx, v_idx = np.unravel_index(min_distance_idx, pairwise_distances.shape)
        u = component_nodes[u_idx]
        v = giant_component_nodes[v_idx]
        
        #Include the edge:
        edges_to_add.append((u, v))
    
        #Add the nodes from this component to the giant component:
        giant_component = pd.concat([giant_component, component])
        giant_component_nodes = np.hstack([giant_component_nodes, component_nodes])
        giant_component_xy = np.vstack([giant_component_xy, component_xy])
        
    #Add the edges to the graph:
    graph_to_return = deepcopy(graph)
    graph_to_return.add_edges_from(edges_to_add)

    return graph_to_return

#####################################################
#GRAPH DRAWING:

def draw_gdf_and_graph(gdf, graph, full_gdf=None, edge_color_att=None, projected_crs=vars._projected_crs, ax=None, figsize=(10,10), edgewidth=0.5):

    node_positions = gdf.to_crs(projected_crs).geometry.centroid
    node_positions_list = list(zip(node_positions.x, node_positions.y))
    node_positions_dict = dict(zip(range(len(node_positions_list)), node_positions_list))

    
    if edge_color_att is not None:
        edge_colors = np.array([graph[u][v][edge_color_att] for u,v in graph.edges()])
        edge_colors = (edge_colors - min(edge_colors))/(max(edge_colors)-min(edge_colors))
    else:
        edge_colors='red'

    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    if full_gdf is None: full_gdf = gdf
    ax = full_gdf.to_crs(projected_crs).plot(ax=ax, color='lightblue', alpha=0.5)
    ax = erase_water(full_gdf).to_crs(projected_crs).plot(ax=ax, color='darkgray')
    ax = erase_water(gdf).to_crs(projected_crs).plot(ax=ax, color='lightgray')
    nx.draw(graph, pos=node_positions_dict, node_color='black', node_size=1, width=edgewidth, edge_color=edge_colors, ax=ax)
    
    return ax

def draw_grid_graph(grid_graph, node_color='black', edge_color_att=None, figsize=(10,10)):
    
    # Get the grid parameters:
    N = grid_graph.order()
    l = int(np.sqrt(N))
    node_positions_dict = {node: (node%l, node//l) for node in range(N)}

    if edge_color_att is not None:
        edge_colors = np.array([grid_graph[u][v][edge_color_att] for u,v in grid_graph.edges()])
        edge_colors = (edge_colors - min(edge_colors))/(max(edge_colors)-min(edge_colors))
    else:
        edge_colors='red'

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw(grid_graph, pos=node_positions_dict, node_color=node_color, node_size=10, width=0.5, edge_color=edge_colors, ax=ax)
    
    return ax

#####################################################