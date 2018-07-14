import csv
import networkx as nx
import matplotlib.pyplot as plt
G=nx.Graph()

from urllib.request import urlretrieve
website = "https://raw.githubusercontent.com/NowakTom/Networkx/master/out_ucidata-gama1.csv"
urlretrieve(website, "out_ucidata-gama1.csv")
f1 = csv.reader(open('out_ucidata-gama1.csv','r'))

website = "https://raw.githubusercontent.com/NowakTom/Networkx/master/out_ucidata-gama2.csv"
urlretrieve(website, "out_ucidata-gama2.csv")
f2 = csv.reader(open('out_ucidata-gama2.csv','r'))

website = "https://raw.githubusercontent.com/NowakTom/Networkx/master/out_ucidata-gama_relation.csv"
urlretrieve(website, "out_ucidata-gama_relation.csv")
f3 = csv.reader(open('out_ucidata-gama_relation.csv','r'))

website = "https://raw.githubusercontent.com/NowakTom/Networkx/master/out_ucidata-gama.csv"
urlretrieve(website, "out_ucidata-gama.csv")	

for row in f1: 
    G.add_nodes_from(row, color = 'red')

for row in f2: 
    G.add_nodes_from(row, color = 'green')

for row in f3:
    if len(row) == 2 : 
        G.add_edge(row[0],row[1])

	
color_map = []

for n in G.nodes():
    color_map.append(G.node[n]['color'])


G = nx.read_weighted_edgelist("out_ucidata-gama.csv", delimiter=",") 
	
#pos=nx.spring_layout(G, dim = 2)
	
#nx.draw_networkx(G, pos, node_color = color_map, with_labels = True, node_size = 500)

#edge_labels=dict([((u,v,),d['weight'])
#             for u,v,d in G.edges(data=True)])
			 
#nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)

nx.draw_circular(G, node_color = color_map, with_labels=True)

import numpy as np

def entropy(X):
     # policz ile razy każda wartość występuje na liście
     # i zamień tę liczbę na prawdopodobieństwo
     _dict = { x: X.count(x)/len(X) for x in X }

     # wylicz entropię zmiennej losowej
     _entropy = -1 * np.sum([p * np.log2(p) for p in _dict.values()])

     return _entropy

lst = [1,2,1,2,1,2,1,2]

print(entropy(lst))

plt.show()