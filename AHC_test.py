import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

def entropy(X):
     _dict = { x: X.count(x)/len(X) for x in X }

     _entropy = -1 * np.sum([p * np.log2(p) for p in _dict.values()])

     return _entropy

	 
def nodes_connected(u, v):
	return u in G.neighbors(v)

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]
   
class Cluster:
	def __init__(self):
		pass
	def __repr__(self):
		return '(%s,%s)' % (self.left, self.right)
	def add(self, clusters, grid, lefti, righti):
		self.left = clusters[lefti]
		self.right = clusters[righti]
		for r in grid:
			r[lefti] = min(r[lefti], r.pop(righti))
		grid[lefti] = list(map(min, zip(grid[lefti], grid.pop(righti))))
		clusters.pop(righti)
		return (clusters, grid)

def agglomerate(labels, grid):
	clusters = labels
	while len(clusters) > 1:
		# find 2 closest clusters
		print(clusters)
		distances = [(1, 0, grid[1][0])]
		for i,row in enumerate(grid[2:]):
			distances += [(i+2, j, c) for j,c in enumerate(row[:i+2])]
		j,i,_ = min(distances, key=lambda x:x[2])
		c = Cluster()
		clusters, grid = c.add(clusters, grid, i, j)
		clusters[i] = c
	return clusters.pop()
   


G=nx.Graph()
summary_list = []
minimum_list =[]
closest_group_list = []



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

lista = [0, 5, 5]

x = 0
y = 1
#for i in group:
#	for j in group:
#		for x in i:
#			for y in j:
#				print(nodes_connected(x, y))


	#group[:] = [v for v in group if v.index != do_usuniecia]
	
	#group.remove(do_usuniecia)
	#do_usuniecia += 1
	

	
	
	
	
wynik = []

	
for node1 in G.nodes:
	waga = 0
	entropia = 0
	odleglosc =  0
	odleglosc_list = []
	for node2 in G.nodes:
		waga = 0
		entropia = 0
		odleglosc =  0
		if nodes_connected(node1, node2) is True:
			waga = G.degree(node1) + G.degree(node2)
			entropia = entropy(G.nodes[node1]['color'] + G.nodes[node2]['color'])
			odleglosc = 1/waga * entropia
		odleglosc_list.append(odleglosc)
	
	wynik.append(odleglosc_list)
	

#print(wynik)
#print(G.nodes)
nody = list(G.nodes)
		

print(agglomerate(nody, wynik))

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