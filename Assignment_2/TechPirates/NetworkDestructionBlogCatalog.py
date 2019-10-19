
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy import stats
import math
from scipy.sparse import csc_matrix
import random
import operator
import scipy.io
import collections
import heapq

import csv
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from math import log
from scipy import stats
from sklearn.decomposition import PCA

# %matplotlib inline
plt.rcParams['figure.figsize'] = 4, 4

Data = open('edges.csv', "r")
Graphtype = nx.Graph()

G = nx.parse_edgelist(Data, comments='t', delimiter=',', create_using=Graphtype,
                      nodetype=int, data=(('weight', float),))



def Degree_Distribution(GA):
    degree_sequence = sorted([d for n, d in GA.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Degree Distribution")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    plt.show() 

def plot_clustering(GA):
  cluster = nx.clustering(GA)
  plt.hist(cluster.values())
  plt.xlabel("Clustering Coeficient")
  plt.ylabel("No of Nodes")
  plt.show()

def get_giant_cluster(GA):
  GC = max(nx.connected_component_subgraphs(GA), key=len)
  print("Nodes in giant cluster = " ,GC.number_of_nodes())
  diameter = nx.algorithms.distance_measures.diameter(GA)
  print("diameter = " , diameter)



#  properties of graph
Degree_Distribution(G)
plot_clustering(G)
get_giant_cluster(G)
print(nx.info(G))
print(nx.number_of_nodes(G))
print(nx.number_of_edges(G))
print(nx.is_directed(G))
print(nx.average_degree_connectivity(G))
for C in nx.connected_component_subgraphs(G):
    print(nx.average_shortest_path_length(C))
yg=[len(c) for c in nx.connected_component_subgraphs(G)]
plt.ylabel("No. of Cluster")
plt.xlabel("No. of nodes")
plt.hist(yg)
print(nx.average_clustering(G))




G_CN = G.copy()
print(nx.info(G_CN))
G_JC = G.copy()
print(nx.info(G_JC))
G_AA = G.copy()
print(nx.info(G_AA))
G_RA = G.copy()
print(nx.info(G_RA))
G_PA = G.copy()
print(nx.info(G_PA))



total_edges=list(G.edges())
methods=['CN','JC','AA','RA','PA']
allm={m:{} for m in methods}



for p in total_edges:
        n1,n2=p
        ngh1 = set(G[n1])
        ngh2 = set(G[n2])
        inter = ngh1.intersection(ngh2) # ngh1 & ngh2
        inter_l = len(inter)
        union_l = len(ngh1.union(ngh2)) #ngh1 | ngh2            
        
        allm['CN'][p]=inter_l
        allm['JC'][p]=(inter_l/union_l) if union_l else 0.0
        allm['AA'][p]=sum([1/log(len(G[z])) for z in inter]) # denom cant be zero as atleast 2 edges
        allm['RA'][p]=sum([1/len(G[z]) for z in inter])
        allm['PA'][p]=len(ngh1)*len(ngh2)


Existing={}
for m in methods:
    Existing[m] = sorted([(e,allm[m][e]) for e in total_edges],key=lambda x:x[1],reverse=True)


# top 30000 edges removed of each method and properties analysed
G_CN.remove_edges_from([i[0] for i in Existing['CN'][:30000]])
for C in nx.connected_component_subgraphs(G_CN):
    print(nx.average_shortest_path_length(C))
print(nx.average_clustering(G_CN))

G_JC.remove_edges_from([i[0] for i in Existing['JC'][:30000]])
for C in nx.connected_component_subgraphs(G_JC):
    print(nx.average_shortest_path_length(C))
print(nx.average_clustering(G_JC))

G_AA.remove_edges_from([i[0] for i in Existing['AA'][:30000]])
for C in nx.connected_component_subgraphs(G_AA):
    print(nx.average_shortest_path_length(C))
print(nx.average_clustering(G_AA))

G_PA.remove_edges_from([i[0] for i in Existing['PA'][:30000]])
for C in nx.connected_component_subgraphs(G_PA):
    print(nx.average_shortest_path_length(C))
print(nx.average_clustering(G_PA))

G_RA.remove_edges_from([i[0] for i in Existing['RA'][:30000]])
for C in nx.connected_component_subgraphs(G_RA):
    print(nx.average_shortest_path_length(C))
print(nx.average_clustering(G_RA))

Degree_Distribution(G_CN)
plot_clustering(G_CN)
get_giant_cluster(G_CN)

Degree_Distribution(G_JC)
plot_clustering(G_JC)
get_giant_cluster(G_JC)

Degree_Distribution(G_AA)
plot_clustering(G_AA)
get_giant_cluster(G_AA)

Degree_Distribution(G_PA)
plot_clustering(G_PA)
get_giant_cluster(G_PA)

Degree_Distribution(G_RA)
plot_clustering(G_RA)
get_giant_cluster(G_RA)

print("number of connected components = " , nx.number_connected_components(G))
print("number of connected components = " , nx.number_connected_components(G_CN))
print("number of connected components = " , nx.number_connected_components(G_JC))
print("number of connected components = " , nx.number_connected_components(G_AA))
print("number of connected components = " , nx.number_connected_components(G_RA))
print("number of connected components = " , nx.number_connected_components(G_PA))

y=[len(c) for c in nx.connected_component_subgraphs(G_JC)]
plt.ylabel("No. of Cluster")
plt.xlabel("No. of nodes")
plt.hist(y)

yg=[len(c) for c in nx.connected_component_subgraphs(G_AA)]
plt.ylabel("No. of Cluster")
plt.xlabel("No. of nodes")
plt.hist(yg)

yg=[len(c) for c in nx.connected_component_subgraphs(G_CN)]
plt.ylabel("No. of Cluster")
plt.xlabel("No. of nodes")
plt.hist(yg)

yg=[len(c) for c in nx.connected_component_subgraphs(G_PA)]
plt.ylabel("No. of Cluster")
plt.xlabel("No. of nodes")
plt.hist(yg)

yg=[len(c) for c in nx.connected_component_subgraphs(G_RA)]
plt.ylabel("No. of Cluster")
plt.xlabel("No. of nodes")
plt.hist(yg)

