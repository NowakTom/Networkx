import networkx as nx
import numpy as np
import doctest
import math
from functools import reduce
import matplotlib.pyplot as plt
from itertools import product, combinations
import sys
from scipy.stats import spearmanr
	
def graph_init(n, p, percent, balance_weight_grp1, balance_weight_grp2):
	'''
	Creating a random graph, coloring nodes, searching cycles and adding attributes 'weight' and 'balance_weight' for edges depending on the function parameters
	:param n: the number of nodes
	:param p: probability for edge creation
	:param percent: percent of positive weight of edges
	:param balance_weight_grp1: positive edge weight value
	:param balance_weight_grp2: negative edge weight value
	'''
	
	#Test graph
	#------------------------------------------------
	graph=nx.Graph()
	graph.add_nodes_from([0,1,2,3,4])
	graph.add_edges_from([(0,1),(0,2), (0,3), (1,2), (1,4)])
	
	#graph = nx.random_graphs.erdos_renyi_graph(n, p)
	
	#positive_edges = math.floor(len(graph.edges) * percent)
	
	#Test color
	#------------------------------------------------
	#negative
	graph[0][1]['balance_weight'] = balance_weight_grp2
	graph[0][3]['balance_weight'] = balance_weight_grp2
	#positive
	graph[0][2]['balance_weight'] = balance_weight_grp1
	graph[1][2]['balance_weight'] = balance_weight_grp1
	graph[1][4]['balance_weight'] = balance_weight_grp1
	
	#for i, unique_combination in enumerate(graph.edges):
	#	if i < positive_edges:
	#		graph[list(unique_combination)[0]][list(unique_combination)[1]]['balance_weight'] = balance_weight_grp1
	#	else:
	#		graph[list(unique_combination)[0]][list(unique_combination)[1]]['balance_weight'] = balance_weight_grp2

	cycles = nx.cycle_basis(graph)
	
	return graph, cycles

def color_graph_nodes(G, color_grp1, color_grp2, weight_grp1, weight_grp2, list_of_colors):
	'''
	Coloring graph nodes
	:param G: input graph
	:param color_grp1: first group nodes color
	:param color_grp2: second group nodes color
	:param weight_grp1: edge weight value that connects nodes with the same color
	:param weight_grp2: edge weight value that connects nodes of different color
	:param list_of_colors: dictionary with the number of the node and the value of its color
	'''
	
	if 'None' in list_of_colors:
		node_colors = {i: np.random.choice([color_grp1, color_grp2]) for i in G.nodes}
	else:
		node_colors = list_of_colors
	
	nx.set_node_attributes(G, node_colors, 'color')
	
	for u, v in G.edges:
		if G.nodes[u]['color'] == G.nodes[v]['color']:
			G[u][v]['weight'] = weight_grp1
		else:
			G[u][v]['weight'] = weight_grp2
	
	return node_colors
	
def entropy(X):
	'''
	Computes the entropy of a collection of values
	:param X: list of values
	:return: Shannon's entropy of `X`
	Tests:
	>>> entropy([1,1,1,1])
	-0.0
	>>> entropy([0,0,1,1])
	1.0
	'''

	_dict = { x: X.count(x)/len(X) for x in X }
	_entropy = -1 * np.sum([p * np.log2(p) for p in _dict.values()])

	return _entropy

def distance(grp1, grp2, G):
	'''
	Computes the custom distance between groups of vertices in graph G
	:param grp1: first group of vertices
	:param grp2: second group of vertices
	:param G: input graph
	:return: distance between groups
	'''

	edges = [(u, v) for u, v in product(grp1, grp2) if (u, v) in G.edges]
	edge_colors = [G.nodes[n]['color'] for n in grp1+grp2]
	sum_edge_weights = sum([G[u][v]['weight'] for (u,v) in edges])

	if not edges:
		return np.inf
	else:
		return (1/sum_edge_weights) * entropy(edge_colors)
		
def frustration_index(G, balance_weight_grp1, balance_weight_grp2):
	'''
	Calculate the frustration coefficient based on the rules:
	1) edge is negative and both its nodes are in the same color
	2) edge is positive and both its nodes are in opposite color
	:param G: input graph
	:param balance_weight_grp1: positive edge weight value
	:param balance_weight_grp2: negative edge weight value
	'''

	frustration  = 0
	for u, v in G.edges:
		if G.nodes[u]['color'] == G.nodes[v]['color'] and G[u][v]['balance_weight'] == balance_weight_grp2:
			frustration += 1
		elif G.nodes[u]['color'] != G.nodes[v]['color'] and G[u][v]['balance_weight'] == balance_weight_grp1:
			frustration += 1
	
	return frustration
	
def indicators(grp, G):
	'''
	Multiplies edges weight for each node in cycles in graph G
	:param grp1: all cycles in graph
	:param G: input graph
	:return: list of indicators for each cycle
	'''

	edges_weight = []
	choosen_nodes = [list(zip(nodes,(nodes[1:]+nodes[:1]))) for nodes in grp] 
	for i, unique_combination in enumerate(choosen_nodes):
		edges_weight = edges_weight + [[G[u][v]['balance_weight'] for (u,v) in choosen_nodes[i]]] 
	indicators =  [reduce(lambda x, y: x*y, edge) for edge in edges_weight] 
	
	return indicators
	
def gini(x):
    '''
	Gini coefficient
	(Warning: This is a concise implementation, but it is O(n**2)
    in time and memory, where n = len(x).  *Don't* pass in huge
    samples!)
	:param x: list of min distances from AHC
	'''

    mad = np.abs(np.subtract.outer(x, x)).mean()
    rmad = mad/np.mean(x)
    g = 0.5 * rmad
	
    return g
	
def dict_color_nodes(nodes, color_grp1, color_grp2):

	'''
	Return a dictionary with the number of the node and the value of its color
	:param nodes: list of indices how to color nodes - 0 = color_grp1, 1 = color_grp2
	:param color_grp1: first group nodes color
	:param color_grp2: second group nodes color
	'''

	colors = {}
	for u, node in enumerate(nodes):
		if node == 0:
			colors.update({u: color_grp1})
		else:
			colors.update({u: color_grp2})

	return colors
		
doctest.testmod()

# ---------- [start] graph initialization --------------------------

n = int(sys.argv[1])
p = 0.9
perc_positive_edges = 0.5

first_grp_color = 'red'
second_grp_color = 'yellow'

same_node_color_edge_weight = 1
opposite_node_color_edge_weight = 2

same_node_color_edge_balance_weight = 1
opposite_node_color_edge_balance_weight = -1

gini_list = []
frustation_list = []
balance_list = []

for number in range(1):
	list_of_colors = ['None']
	
	G, cycles = graph_init(n, p, perc_positive_edges, same_node_color_edge_balance_weight, opposite_node_color_edge_balance_weight)
	node_colors = color_graph_nodes(G, first_grp_color, second_grp_color, same_node_color_edge_weight, opposite_node_color_edge_weight, list_of_colors)
	
	# ---------- [end] graph initialization --------------------------
	
	min_distances = []
	indices_of_frustration = []
	tuples_to_lists = lambda lst: [list(unique_combination) for unique_combination in lst]
	flatten = lambda lst: [unique_combination for sublst in lst for unique_combination in sublst]
	
	unique_combinations = tuples_to_lists(list(product(range(2), repeat=len(G.nodes()))))
	
	for i, unique_combination in enumerate(unique_combinations):
	
		colors = dict_color_nodes(unique_combination, first_grp_color, second_grp_color)
	
		nx.set_node_attributes(G, colors, 'color')
		indices_of_frustration.append(frustration_index(G, same_node_color_edge_balance_weight, opposite_node_color_edge_balance_weight))
		
	lowest_frustration_index_with_colors = unique_combinations[np.argmin(indices_of_frustration)]
	lowest_frustration = indices_of_frustration[np.argmin(indices_of_frustration)]
	
	list_of_colors = dict_color_nodes(lowest_frustration_index_with_colors, first_grp_color, second_grp_color)
	
	#---------- [start] reprint nodes with param: list_of_colors --------------------------
	
	node_colors = color_graph_nodes(G, first_grp_color, second_grp_color, same_node_color_edge_weight, opposite_node_color_edge_weight, list_of_colors)
	
	#---------- [start] reprint nodes with param: list_of_colors --------------------------
	
	groups = [[nodes] for nodes in G.nodes()]
	print('AHC:')
	while len(groups) > 1:
	
		pairs = tuples_to_lists(combinations(groups, 2))
		distances = [distance(u, v, G) for u,v in pairs]
	
		min_distance = min(distances)
		min_pair_idx = distances.index(min_distance)
		min_distances.append(min_distance)
	
		groups.remove(pairs[min_pair_idx][0])
		groups.remove(pairs[min_pair_idx][1])
		groups.append(flatten(pairs[min_pair_idx]))
	
		print(groups, 'minimum distance: ', min_distance)
	
	cycles_indicators = indicators(cycles, G)
	balance_index = sum(cycles_indicators)
	
	
	print('list of minimum distances: ', min_distances)
	print('Gini coefficient: ', gini(min_distances))
	print('\n#########################################\n')
	print('Balance index:')
	print('graph cycles: ', cycles)
	print('cycles indicators: ', cycles_indicators)
	print('graph balance index: ', balance_index)
	print('\n#########################################\n')
	print('Frustration index:')
	print('unique combinations for frustration: ', unique_combinations)
	print('indices of frustration: ', indices_of_frustration)
	print('lowest_frustration_index_with_colors (0 means first color, 1 means second):', lowest_frustration_index_with_colors)
	print('lowest frustration index: ', lowest_frustration)
	print('\n#########################################\n')
	
	gini_list.append(gini(min_distances))
	frustation_list.append(lowest_frustration)
	balance_list.append(balance_index)

#print('Spearman correlation coefficients between gini and frustration index: ')
#print(spearmanr(gini_list, frustation_list))
#print('\n')
#print('Spearman correlation coefficients between gini and balance index: ')
#print(spearmanr(gini_list, balance_list))

nx.draw(G, with_labels=True, node_color=list(node_colors.values()))
plt.show()