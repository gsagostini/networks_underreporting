import networkx as nx
import numpy as np
import pickle as pkl
import time
start_time = time.time()

import sys
sys.path.append('../../d03_src/')
import model_MCMC
import process_graph as prg

from multiprocessing import Pool, cpu_count
import argparse

##########################################################################

parser = argparse.ArgumentParser()

#System outputs:
parser.add_argument("--results_folder", type=str, default='/share/garg/gs665/networks_underreporting/d05_joboutputs/generation/')

#Structural parameters:
parser.add_argument("--graph_type", type=str, default='census')
parser.add_argument("--grid_side_length", type=int, default=50)
parser.add_argument("--census_unit", type=str, default='tracts')
parser.add_argument("--geohash_precision", type=int, default=6)

#Configuring the graph:
parser.add_argument('--trim_graph_variables', nargs='+', default=['population', 'park_area', 'land_area'])
#Available trimming parameters are: ['population', 'park_area', 'land_area', 'edge_length', 'node_degree']
parser.add_argument("--max_degree", type=int, default=9)
parser.add_argument("--max_edgelength", type=float, default=2000) #meters
parser.add_argument("--min_pop", type=int, default=100)
parser.add_argument("--max_parkarea", type=float, default=0.75)
parser.add_argument("--min_landarea", type=float, default=1/3)

#MCMC hyperparameters:
parser.add_argument("--MCMC_n_burnin", type=int, default=1_000)

#Sample size:
parser.add_argument("--sample_size", type=int, default=500)

args = parser.parse_args()

##########################################################################
#COLLECT POSSIBLE PARAMETER VALUES:

possible_theta_0 = [round(0.05*k,2) for k in range(-10, 11)]
possible_theta_1 = [round(0.05*k,2) for k in range(-10, 11)]

##########################################################################
#GENERATE THE GRAPH:

assert args.graph_type in ['census', 'grid', 'geohash']

#Grid graph:
if args.graph_type == 'grid':
    print(f'Using a grid graph of side {args.grid_side_length}')
    graph = prg.generate_graph_grid(graph_grid_l)

#Census graph:
elif args.graph_type == 'census':
    assert args.census_unit in ['tracts', 'block_groups', 'blocks']
    print(f"Using a census {args.census_unit} graph")
    print(f"Parameters to trim by: {', '.join(args.trim_graph_variables)}")
    if 'population' in args.trim_graph_variables: print(f"Minimum population: {args.min_pop}")
    if 'park_area' in args.trim_graph_variables: print(f"Maximum park area (fraction): {args.max_parkarea}")
    if 'land_area' in args.trim_graph_variables: print(f"Maximum land area (fraction): {1-args.min_landarea}")
    
    _, graph = prg.generate_graph_census(census_unit=args.census_unit,
                                         state='NY',
                                         counties=['New York','Bronx','Kings','Queens','Richmond'],
                                         weights_type='rook',
                                         remove_water=True if 'land_area' in args.trim_graph_variables else False,
                                         remove_zeropop=True if 'population' in args.trim_graph_variables else False,
                                         remove_parks=True if 'park_area' in args.trim_graph_variables else False,
                                         tresh_population=args.min_pop,
                                         tresh_parkarea=args.max_parkarea,
                                         enforce_connection=True)

#Geohash graph:
elif args.graph_type == 'geohash':
    assert args.geohash_precision in [5, 6, 7]
    print(f"Using a geohash with precision {args.geohash_precision} graph")
    print(f"Parameters to trim by: {', '.join(args.trim_graph_variables)}")   
    if 'population' in args.trim_graph_variables: print(f"Minimum population: {args.min_pop}")
    if 'park_area' in args.trim_graph_variables: print(f"Maximum park area (fraction): {args.max_parkarea}")
    if 'land_area' in args.trim_graph_variables: print(f"Maximum land area (fraction): {1-args.min_landarea}")
        
    _, graph = prg.generate_graph_geohash(precision=args.geohash_precision,
                                          weights_type='rook',
                                          remove_water=True if 'land_area' in args.trim_graph_variables else False,
                                          remove_zeropop=True if 'population' in args.trim_graph_variables else False,
                                          remove_parks=True if 'park_area' in args.trim_graph_variables else False,
                                          tresh_water=args.min_landarea,
                                          tresh_population=args.min_pop,
                                          tresh_parkarea=args.max_parkarea,
                                          enforce_connection=True)
run_time = time.time() - start_time
print(f"Graph runtime: {(run_time//3600):.0f} hours, {((run_time%3600)//60):.0f} minutes and {(run_time%60):.0f} seconds")

##########################################################################

def parameter_generator(possible_theta_0,
                        possible_theta_1,
                        graph=graph,
                        n_burnin=200,
                        sample_size=500):
    
    for theta0 in possible_theta_0:
        for theta1 in possible_theta_1:
            yield {'theta0':theta0, 'theta1':theta1, 'graph':graph, 'burnin':n_burnin, 'sample_size':sample_size}
            
##########################################################################

def generate_data(graph, theta0, theta1, psi=1., burnin=10_000):
    A_list, T_list, _ = model_MCMC.run_MCMC(graph,
                                            fixed_params=True,
                                            MCMC_iterations=burnin,
                                            params={'psi': psi, 'theta0': theta0, 'theta1': theta1})
    return A_list[-1], T_list[-1]

def simulator(params):
    A_arr = []
    for iteration in range(params['sample_size']):
        A, T = generate_data(params['graph'], params['theta0'], params['theta1'], burnin=params['burnin'])
        A_arr.append(A)    
    return params['theta0'], params['theta1'], np.array(A_arr)
    
##########################################################################
    
#Create the Pool object:
pool = Pool(None)

#Generate the parameters:
generator = parameter_generator(possible_theta_0,
                                possible_theta_1,
                                graph=graph,
                                n_burnin=args.MCMC_n_burnin,
                                sample_size=args.sample_size)

#Iterate:
A_arr_dict = {}
for result in pool.imap_unordered(simulator, generator):
    theta0, theta1, A = result
    A_arr_dict[(theta0, theta1)] = A
    pool.close()

##########################################################################
    
#Define a name for the graph:
if args.graph_type == 'grid': graph_descriptor = args.grid_l
if args.graph_type == 'census': graph_descriptor = args.census_unit
if args.graph_type == 'geohash': graph_descriptor = args.geohash_precision

#Save:
filename = f'generated_A_{args.sample_size}samples_{args.graph_type}_{graph_descriptor}'
with open(f'{args.results_folder}{filename}.pkl', 'wb') as handle:
    pkl.dump(A_arr_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
##########################################################################

run_time = time.time() - start_time
print(f'\nTotal runtime: {(run_time//3600):.0f} hours, {((run_time%3600)//60):.0f} minutes and {(run_time%60):.0f} seconds')