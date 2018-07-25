import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

def entropy(X):
     _dict = { x: X.count(x)/len(X) for x in X }

     _entropy = -1 * np.sum([p * np.log2(p) for p in _dict.values()])

     return _entropy

G=nx.Graph()
summary_list = []


G = nx.random_graphs.erdos_renyi_graph(31,0.3)

node_colors = { i: np.random.choice(['white','black']) for i in G.nodes}

nx.set_node_attributes(G, node_colors, 'color')

for u,v in G.edges:	
	if  G.nodes[u]['color'] ==  G.nodes[v]['color']:
		G[u][v]['weight']= 1
	else:
		G[u][v]['weight']= 2

group = [[i] for i in G]

#while (len(group) >  1):
#	for i in group:
#		for u,v in G.edges:	
#			if  G.nodes[u]['color'] !=  G.nodes[v]['color']:
				




#for node in G:	
#	print(G.nodes[node]['color'])

print(group)
#print(entropy(G.nodes[i]['color']))

#for i in G.nodes:	
#	color_list.append(G.nodes[i]['color'])

#1/x * entropy(dict_colors


#print(entropy(list(node_colors.values())))
#print(len(group))
#print(sum(G.degree(weight = 'weight')))
#Z = linkage(X, 'single')
#print(color_list)
#print(entropy(color_list))
#print(G.degree(weight = 'weight')[0])
#print(len(G.degree))
#print(len(color_list))

#print(G.degree(weight = 'weight')[0])

X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]
Z = linkage(X, 'single')
print(X)
print(Z)
print(type(Z))