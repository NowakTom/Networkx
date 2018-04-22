import csv
import networkx as nx
import matplotlib.pyplot as plt
g=nx.Graph()
from urllib.request import urlretrieve
website = "https://raw.githubusercontent.com/NowakTom/Networkx/master/out_ucidata-gama.txt"
urlretrieve(website, "out_ucidata-gama.txt")

f1 = csv.reader(open("out_ucidata-gama.txt","r"))
for row in f1: G.add_nodes_from(row[0], color = 'blue')

for row in f1: G.add_nodes_from(row[1], color = 'red')

for row in f1: G.add_edge(row[0],row[1])

color_map = []

for n in G.nodes(): color_map.append(G.node[n]['color']) nx.draw_networkx(G, node_color = color_map, with_labels = True, node_size = 500)

