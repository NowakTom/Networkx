import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

def entropy(X):
     _dict = { x: X.count(x)/len(X) for x in X }

     _entropy = -1 * np.sum([p * np.log2(p) for p in _dict.values()])

     return _entropy

	 
def nodes_connected(u, v):
	return u in G1.neighbors(v)

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

G=nx.Graph()
summary_list = []
lista_u =[]
lista_v = []



G = nx.random_graphs.erdos_renyi_graph(31,0.3)

node_colors = { i: np.random.choice(['white','black']) for i in G.nodes}

nx.set_node_attributes(G, node_colors, 'color')

for u,v in G.edges:	
	if  G.nodes[u]['color'] ==  G.nodes[v]['color']:
		G[u][v]['weight']= 1
	else:
		G[u][v]['weight']= 2

group = [[i] for i in G]

G1 = nx.Graph(G)

for node1 in G1.nodes:
	najmniejsza = 100
	najblizszy_wezel_2 = 99
	for node2 in G1.nodes:
		waga = float()
		entropia = float()
		odleglosc =  float()
		if nodes_connected(node1, node2) is True:
			waga = G1.degree(node1) + G1.degree(node2)
			entropia = entropy(G.nodes[node1]['color'] + G.nodes[node2]['color'])
		if waga > 0:
			odleglosc = 1/waga * entropia
		if odleglosc > 0 and odleglosc < najmniejsza:
			najmniejsza = odleglosc
			najblizszy_wezel_2 = node2
	print(najmniejsza)
	print(node1)
	print(najblizszy_wezel_2)
#		summary_list.append(odleglosc)
		


#		summary_list.append(G1.degree(node))
#while (len(group) >  1):
#	for i in group:
#		for u,v in G.edges:	
#			if  G.nodes[u]['color'] !=  G.nodes[v]['color']:
				


#summary_list = remove_values_from_list(summary_list, 0)
#print(min(summary_list))


#print(list(G.neighbors(0)))

#for node in G:	
#	print(G.nodes[node]['color'])



#print(G.nodes[30]['color'])
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

#X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]
#Z = linkage(X, 'single')
#print(X)
#print(Z)
#print(type(Z))
#dn = dendrogram(Z)
#plt.show()