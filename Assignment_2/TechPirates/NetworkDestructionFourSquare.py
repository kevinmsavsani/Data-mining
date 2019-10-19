
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
subSize=10000




def load_restaurant_dataset():
    path = 'dataset_ubicomp2013_checkins.txt'
    infile = open(path, 'r')
    a = set()
    b = set()
    edges = []
    for line in infile:
        s=line.strip().split(None)
        u=-1*int(s.pop(0)) -10
        v=int(s.pop(0))
        a.add(u)
        b.add(v)
        edges.append((u,v))
    top_nodes = {}
    bottom_nodes = {}
    count = 0 
    for x in a:
        top_nodes[x] = count
        count = count + 1
    count  = 0    
    for y in b:
        bottom_nodes[y] = count
        count  = count + 1
    
    A = np.zeros((len(a),len(b)))
    for edge in edges:
        e1 = top_nodes[edge[0]]
        e2 = bottom_nodes[edge[1]]
        A[e1, e2] = 1
    
    A = np.dot(A,A.T)
    for i in range(0,A.shape[0]):
        for j in range(0,A.shape[1]):
            if i == j :
                A[i,j] = 0
            else:
                if A[i,j] != 0:
                  A[i,j] = 1
                else:
                  A[i,j] = 0
          
                                     
    G=nx.from_numpy_matrix(A)
    return G
G = load_restaurant_dataset()




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
G_JC = G.copy()
G_AA = G.copy()
G_RA = G.copy()
G_PA = G.copy()
G_RP = G.copy()


total_edges=list(G.edges())
methods=['CN','JC','AA','RA','PA','RP']
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




def rooted_pagerank(g, node, d = 0.85, epsilon = 1e-4):
    ordered_nodes = sorted(g.nodes())
    root = ordered_nodes.index(node)
    adjecancy = nx.to_numpy_array(g, nodelist = ordered_nodes)
    m = np.copy(adjecancy)

    n = len(g)

    for i in range(len(g)):
        row_norm = np.linalg.norm(m[i], ord = 1)
        if row_norm != 0:
            m[i] = m[i] / row_norm

    m = m.transpose()

    rootvec = np.zeros(len(g))
    rootvec[root] = 1

    vect = np.random.rand(n)
    vect = vect / np.linalg.norm(vect, ord = 1)
    last_vect = np.ones(n) * 100 # to ensure that does not hit epsilon randomly in first step

    iterations = 0
    while np.linalg.norm(vect - last_vect, ord = 2) > epsilon:
        last_vect = vect.copy()
        vect = d * np.matmul(m, vect) + (1 - d) * rootvec
        iterations += 1

    eigenvector = vect / np.linalg.norm(vect, ord = 1)

    eigen_dict = {}
    for i in range(len(ordered_nodes)):
        eigen_dict[ordered_nodes[i]] = eigenvector[i]

    return eigen_dict

for g in G.nodes():
        aa = rooted_pagerank(G, g)
        for v in G.neighbors(g):
          p = (g,v)
          allm['RP'][p] = aa[v]



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

G_RP.remove_edges_from([i[0] for i in Existing['RP'][:30000]])
for C in nx.connected_component_subgraphs(G_RP):
    print(nx.average_shortest_path_length(C))
print(nx.average_clustering(G_RP))

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

Degree_Distribution(G_RP)
plot_clustering(G_RP)
get_giant_cluster(G_RP)

print("number of connected components = " , nx.number_connected_components(G))
print("number of connected components = " , nx.number_connected_components(G_CN))
print("number of connected components = " , nx.number_connected_components(G_JC))
print("number of connected components = " , nx.number_connected_components(G_AA))
print("number of connected components = " , nx.number_connected_components(G_RA))
print("number of connected components = " , nx.number_connected_components(G_PA))
print("number of connected components = " , nx.number_connected_components(G_RP))


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

yg=[len(c) for c in nx.connected_component_subgraphs(G_RP)]
plt.ylabel("No. of Cluster")
plt.xlabel("No. of nodes")
plt.hist(yg)
