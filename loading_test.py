import csv
import networkx as nx
import matplotlib.pyplot as plt
g=nx.Graph()
from urllib.request import urlretrieve
website = "https://raw.githubusercontent.com/NowakTom/Networkx/master/out_ucidata-gama.txt"
urlretrieve(website, "out_ucidata-gama.txt")

