import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

def entropy(X):
     _dict = { x: X.count(x)/len(X) for x in X }

     _entropy = -1 * np.sum([p * np.log2(p) for p in _dict.values()])

     return _entropy

G=nx.Graph()
#G1=nx.Graph()
#G2=nx.Graph()
dict_colors = []
#_dict2 = []
#nodes_colors = []
edges_weights = []
d_G1_G2 = []

G = nx.random_graphs.erdos_renyi_graph(31,0.3)

#for node in G:
#	if node % 3 == 0:
#		G1.add_node(node, color = 'white')
#	else:
#		G2.add_node(node, color = 'black')

#G1_color=nx.get_node_attributes(G1,'color')
#G2_color=nx.get_node_attributes(G2,'color')	

#for node in G1:
#		dict_colors.append('white')

#for node in G2:
#		_dict2.append('black')

#nodes_colors = dict_colors + _dict2

#for node in G:
#	if node % 3 == 0:
#		G1.add_node(node, color = 'white')
#	else:
#		G2.add_node(node, color = 'black')

edges_weights_summary = np.zeros(len(G))

for node in G:
	if node % 3 == 0:
		dict_colors.append('white')
	else:
		dict_colors.append('black')

for u,v in G.edges:	
	if dict_colors[u] == dict_colors[v]:
		edges_weights.append(1)
	else:
		edges_weights.append(2)
		
for u,v in G.edges:	
	if u == u:
		edges_weights_summary[u] = edges_weights_summary[u] + edges_weights[u]

for x in edges_weights_summary:	
	if x != 0:
		d_G1_G2.append(1/x * entropy(dict_colors))
	else:
		d_G1_G2.append(0)

#d_G1_G2 = 1/sum(edges_weights) * entropy(dict_colors)
		
print("Graph nodes")
print(list(G.nodes()))
print("Nodes colors")
print(dict_colors)
print("Entropy")
print(entropy(dict_colors))
print("Edges weights")
print(edges_weights)
print("Nodes - edges weights")
print(edges_weights_summary)
print("Nodes distance")
print(d_G1_G2)

X = [[i] for i in d_G1_G2]

Z = linkage(X, 'single')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)
plt.show()