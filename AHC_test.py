import networkx as nx
import numpy as np
import doctest
import math
from functools import reduce

import matplotlib.pyplot as plt

from itertools import product, combinations

def graph_init(n, p, percent, color_grp1, color_grp2, weight_grp1, weight_grp2, balance_weight_grp1, balance_weight_grp2):
	'''
	Creating a random graph, coloring nodes, searching cycles and adding attributes 'weight' and 'balance_weight' for edges depending on the function parameters
	:param n: the number of nodes
	:param p: probability for edge creation
	:param percent: percent of positive weight of edges
	:param color_grp1: first group nodes color
	:param color_grp2: second group nodes color
	:param weight_grp1: edge weight value that connects nodes with the same color
	:param weight_grp2: edge weight value that connects nodes of different color
	:param balance_weight_grp1: positive edge weight value
	:param balance_weight_grp2: negative edge weight value
	'''
	
	graph = nx.random_graphs.erdos_renyi_graph(n, p)
	
	node_colors = {i: np.random.choice([color_grp1, color_grp2]) for i in graph.nodes}

	nx.set_node_attributes(graph, node_colors, 'color')
	
	for u, v in graph.edges:
		if graph.nodes[u]['color'] == graph.nodes[v]['color']:
			graph[u][v]['weight'] = weight_grp1
		else:
			graph[u][v]['weight'] = weight_grp2
			
	positive_edges = math.floor(len(graph.edges) * percent)
	
	
	for i, unique_combination in enumerate(graph.edges):
		if i < positive_edges:
			graph[list(unique_combination)[0]][list(unique_combination)[1]]['balance_weight'] = balance_weight_grp1
		else:
			graph[list(unique_combination)[0]][list(unique_combination)[1]]['balance_weight'] = balance_weight_grp2

	cycles = nx.cycle_basis(graph)
	
	return graph, node_colors, cycles

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
	choosen_nodes = [list(zip(nodes,(nodes[1:]+nodes[:1]))) for nodes in grp] #wyszukanie każdej pary w cyklu tak aby poprawnie móc wybrać wagę krawędzi między tymi dwoma węzłami
	for i, unique_combination in enumerate(choosen_nodes):
		edges_weight = edges_weight + [[G[u][v]['balance_weight'] for (u,v) in choosen_nodes[i]]] #dla każdej pary wierchołków wyszukanie łączącej ich krawędzi i pobranie dla niej wagi (z parametru balance_weight)
	indicators =  [reduce(lambda x, y: x*y, edge) for edge in edges_weight] #mnożenie wag krawędzi w każdym cyklu
	
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
		
doctest.testmod()

# ---------- [start] graph initialization --------------------------

n = 3
p = 0.5
perc_positive_edges = 0.3

first_grp_color = 'red'
second_grp_color = 'yellow'

same_node_color_edge_weight = 1
opposite_node_color_edge_weight = 2

same_node_color_edge_balance_weight = 1
opposite_node_color_edge_balance_weight = -1

G, node_colors, cycles = graph_init(n, p, perc_positive_edges, first_grp_color, second_grp_color, same_node_color_edge_weight, opposite_node_color_edge_weight, same_node_color_edge_balance_weight, opposite_node_color_edge_balance_weight)

# ---------- [end] graph initialization --------------------------

min_distances = []
indices_of_frustration = []
tuples_to_lists = lambda lst: [list(unique_combination) for unique_combination in lst]
flatten = lambda lst: [unique_combination for sublst in lst for unique_combination in sublst]

groups = [[nodes] for nodes in G.nodes()]

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
unique_combinations = tuples_to_lists(list(product(range(2), repeat=len(G.nodes()))))

for i, unique_combination in enumerate(unique_combinations):
	kolory = {}
	for u, node in enumerate(unique_combination):
		if node == 0:
			kolory.update({u : first_grp_color})
		else:
			kolory.update({u : second_grp_color})

	nx.set_node_attributes(G, kolory, 'color')
	indices_of_frustration.append(frustration_index(G, same_node_color_edge_balance_weight, opposite_node_color_edge_balance_weight))
	labels = nx.get_edge_attributes(G,'balance_weight')

print('list of minimum distances: ', min_distances)
print('graph cycles: ', cycles)
print('cycles indicators: ', cycles_indicators)
print('graph balance index: ', balance_index)
print('Gini coefficient: ', gini(min_distances))
print('indices of frustration: ', indices_of_frustration)

nx.draw(G, with_labels=True, node_color=list(node_colors.values()))
plt.show()
