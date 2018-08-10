import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

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

def entropy(X):
     _dict = { x: X.count(x)/len(X) for x in X }

     _entropy = -1 * np.sum([p * np.log2(p) for p in _dict.values()])

     return _entropy

	 
def nodes_connected(u, v):
	return u in G.neighbors(v)

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]
   
G=nx.Graph()

G = nx.random_graphs.erdos_renyi_graph(32,0.3)

node_colors = { i: np.random.choice(['white','black']) for i in G.nodes}

nx.set_node_attributes(G, node_colors, 'color')

for u,v in G.edges:	
	if  G.nodes[u]['color'] ==  G.nodes[v]['color']:
		G[u][v]['weight']= 1
	else:
		G[u][v]['weight']= 2


G1 = nx.Graph(G)

#lista = [(0, 5), 5, 7]

node_list1 = list(G.nodes)
node_list2 = list(G.nodes)
	
while len(node_list1) > 1:
	grid = []
	for node1 in node_list1:
		v_distance_list = []
		for node2 in node_list2:
			v_weight = 0
			v_entropy = 0
			v_distance =  0
			connected = False
			color_list = []
			if isinstance(node1, (tuple,)) is True: #sprawdzenie czy element listy node_list1 jest grupa typu tuple
				for e in node1:
					if nodes_connected(e, node2) is True: #sprawdzenie czy grupy sa polaczone
						connected = True
						v_weight = v_weight + G.degree(e) #suma wag polaczonych grup
						color_list.append(G.nodes[e]['color']) #sumaryczna lista kolorow polaczonych wezlow w grupach
				if connected is True: #jesli istnieje polaczenie w grupie, licz odleglosc, inaczej odleglosc = 0
					v_weight = v_weight + G.degree(node2)
					v_entropy = entropy(color_list + G.nodes[node2]['color'])
					v_distance = 1/v_weight * v_entropy
			else:
				if nodes_connected(node1, node2) is True:
					v_weight = G.degree(node1) + G.degree(node2)
					v_entropy = entropy(G.nodes[node1]['color'] + G.nodes[node2]['color'])
					v_distance = 1/v_weight * v_entropy
			v_distance_list.append(v_distance)
		grid.append(v_distance_list) #macierz odleglosci
	#poszukiwania najblizszych grup
	print(node_list1)
	distances = [(1, 0, grid[1][0])]
	for i,row in enumerate(grid[2:]):
		distances += [(i+2, j, c) for j,c in enumerate(row[:i+2])]
	j,i,_ = min(distances, key=lambda x:x[2])
	c = Cluster()
	node_list1, grid = c.add(node_list1, grid, i, j)
	node_list1[i] = c
	node_list1.pop()
	node_list2.pop()
	
