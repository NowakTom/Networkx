import networkx as nx
import numpy as np
import doctest
from functools import reduce

import matplotlib.pyplot as plt

from itertools import product, combinations

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

def cycles_index(grp, G):
	'''
	Multiplies edges weight for each node in cycles in graph G
	:param grp1: all cycles in graph
	:param G: input graph
	:return: index for each cycle
	'''

	i = 0
	edges_weight = []
	choosen_nodes = [list(zip(nodes,(nodes[1:]+nodes[:1]))) for nodes in grp] #wyszukanie każdej pary w cyklu tak aby poprawnie móc wybrać wagę krawędzi między tymi dwoma węzłami
	while i < len(choosen_nodes):
		edges_weight = edges_weight + [[G[u][v]['balance_weight'] for (u,v) in choosen_nodes[i]]] #dla każdej pary wierchołków wyszukanie łączącej ich krawędzi i pobranie dla niej wagi (z parametru balance_weight)
		i = i + 1
	cycles_index =  [reduce(lambda x, y: x*y, edge) for edge in edges_weight] #mnożenie wag krawędzi w każdym cyklu
	
	return cycles_index

		
doctest.testmod()

# ---------- [start] graph initialization --------------------------

G = nx.random_graphs.erdos_renyi_graph(10, 0.3)

node_colors = {i: np.random.choice(['red', 'yellow']) for i in G.nodes}

nx.set_node_attributes(G, node_colors, 'color')

for u, v in G.edges:
	if G.nodes[u]['color'] == G.nodes[v]['color']:
		G[u][v]['weight'] = 1
		G[u][v]['balance_weight'] = 1 #dodanie nowego parametru ze względu na to, iż w przypadku wartości ujemnej parametr sum_edge_weights wyliczany jest jako zero, a wyliczanie w ramach funkcji odległości wywraca się na dzieleniu przez zero
	else:
		G[u][v]['weight'] = 2
		G[u][v]['balance_weight'] = -1
		
cycles = nx.cycle_basis(G)

# ---------- [end] graph initialization --------------------------

tuples_to_lists = lambda lst: [list(elem) for elem in lst]
flatten = lambda lst: [elem for sublst in lst for elem in sublst]

groups = [[nodes] for nodes in G.nodes()]

while len(groups) > 1:

	pairs = tuples_to_lists(combinations(groups, 2))
	distances = [distance(u, v, G) for u,v in pairs]

	min_distance = min(distances)
	min_pair_idx = distances.index(min_distance)

	groups.remove(pairs[min_pair_idx][0])
	groups.remove(pairs[min_pair_idx][1])
	groups.append(flatten(pairs[min_pair_idx]))

	print(groups, 'minimum distance: ', min_distance)

cycles_indicators = cycles_index(cycles, G)
balance_index = sum(cycles_indicators)

print('graph cycles: ', cycles)
print('cycles indicators: ', cycles_indicators)
print('graph balance index: ', balance_index)

nx.draw(G, with_labels=True, node_color=list(node_colors.values()))
plt.show()


