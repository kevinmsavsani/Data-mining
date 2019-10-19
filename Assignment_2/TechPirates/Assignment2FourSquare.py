
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
import seaborn as sns
import pandas as pd
import csv
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from math import log
from scipy import stats
from sklearn.decomposition import PCA

import math
from sklearn import tree
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = 4, 4
	

## PREPROCESSING
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

G.remove_edges_from(G.selfloop_edges())


total_edges=list(G.edges())
np.random.shuffle(total_edges)
l=int(len(total_edges)*0.7)
edges_0,ETEs = total_edges[:l], total_edges[l:]

"""
 Use all the edges present in the network as "YES", and randomly choose equal number of "Non-existing" edges as "NO". 
"""
# Randomly choose equal sized (entire)fold of NO edges
nonETEs=random.sample(list(nx.non_edges(G)),len(ETEs))
total_edges=ETEs+nonETEs #total in consideration

xe,nxe=len(ETEs),len(nonETEs)
methods=['CN','JC','AA','RA','PA']

N=G.number_of_nodes()
nodelist = list(G.nodes())

print(nx.info(G))
print(nx.number_of_nodes(G))
print(nx.number_of_edges(G))
print(nx.is_directed(G))


def nonedges(G,u):  #a generator with (u,v) for every non neighbor v
    for v in nx.non_neighbors(G, u):
        yield (u, v)


for u in G.nodes():
    adar = nx.adamic_adar_index(G,nonedges(G,u))
    for v in nx.non_neighbors(G, u):
        com = nx.common_neighbors(G,u,v)
    jac = nx.jaccard_coefficient(G,nonedges(G,u))
    res = nx.resource_allocation_index(G,nonedges(G,u))
    pre = nx.preferential_attachment(G,nonedges(G,u))


allm={m:{} for m in methods}
toponame="datasetFourSquare"+".csv"
with open(toponame,"w",newline="", encoding="utf-8") as f:   # binary mode for windows \r\n prob
    writer = csv.writer(f, delimiter=',')        
    writer.writerow(['node1','node2']+methods+['label'])

    for p in total_edges:
            n1,n2=p
            ngh1 = set(G[n1])
            ngh2 = set(G[n2])
            inter = ngh1.intersection(ngh2) # ngh1 & ngh2
            inter_l = len(inter)
            union_l = len(ngh1.union(ngh2)) #ngh1 | ngh2            
            
            allm['CN'][p]=inter_l
            allm['JC'][p]=(inter_l/union_l) if union_l else 0.0
            allm['AA'][p]=sum([1/log(len(G[z])) for z in inter]) 
            allm['RA'][p]=sum([1/len(G[z]) for z in inter])
            allm['PA'][p]=len(ngh1)*len(ngh2)
            
            writer.writerow(list(p) + [allm[m][p] for m in methods] + [1 if G.has_edge(n1,n2) else 0])




df=pd.read_csv(toponame)
df=df[methods]
sns.pairplot(df)
plt.show()

for m in methods:
    plt.subplots()
    plt.xlabel(m)
    df[m].hist(align='mid',bins=30)


def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(axis='both')
    labels=['X']+methods
    ax1.set_xticklabels(labels,fontsize=26)
    ax1.set_yticklabels(labels,fontsize=26)
    fig.colorbar(cax, ticks=list([x/100 for x in range(0,100,5)]))
    plt.show()

print(df.corr())
correlation_matrix(df)


#precision and recall for topological algorithms#
def precision_recall(score_labels, k):
    l=len(score_labels)
    thresh = score_labels[int(l*k)][0]           #all above threshold will have greater cn/jj/aa value. so, label should be 1
    truepos = 0
    falsepos = 0
    falseneg=0
    trueneg =0
    for i in range(0, l):
        if(score_labels[i][1]==1 and score_labels[i][0]>=thresh):
            truepos+=1
        elif (score_labels[i][1]==0 and score_labels[i][0]>=thresh):
            falsepos+=1
        if(score_labels[i][1]==0 and score_labels[i][0]<=thresh):
            trueneg+=1
        elif(score_labels[i][1]==1 and score_labels[i][0]<=thresh):
            falseneg+=1
    
    truepos = float(truepos)
    trueneg = float(trueneg)
    falsepos = float(falsepos)
    falseneg = float(falseneg)
    p = '%.3f'%float(truepos/(truepos+falsepos))
    r = '%.3f'%float(truepos/(truepos+falseneg))
    return [p,r]

all_edges_score,nonExisting={},{}
for m in methods:
    nonExisting[m] = sorted([(e,allm[m][e]) for e in nonETEs],key=lambda x:x[1])
    all_edges_score[m] = sorted([(allm[m][e],1) for e in ETEs]+[(allm[m][e],0) for e in nonETEs],key=lambda x:x[0],reverse=True)


topoaucname="topological_metricsFourSquare"+".csv"

with open(topoaucname,"w",newline="", encoding="utf-8") as f:   # binary mode for windows \r\n prob
    writer = csv.writer(f, delimiter=',')        
    writer.writerow(['node1','node2']+methods)
    sumAUC={m:0 for m in methods}
    total={m:0 for m in methods}
    for p in ETEs:
        row=[p[0],p[1]]
        
        for m in methods:        
            score=allm[m][p]
            count=0.0
            for s in nonExisting[m]:
                if(s[1]<score):
                    count+=1.0
                elif(s[1]==score):
                    count+=0.5
                else:
                    break
            local_auc = count/nxe
            sumAUC[m]+=local_auc            
            total[m]+=1
            row.append(local_auc)
            
        writer.writerow(row)
    row=['overall AUC','-']
    row2=['PR@25%','-']
    row3=['PR@50%','-']
    row4=['PR@75%','-']
    for m in methods:
        assert(total[m]==xe)
        row.append(sumAUC[m]/xe) # avg 
        row2.append(precision_recall(all_edges_score[m],k=0.25))
        row3.append(precision_recall(all_edges_score[m],k=0.50))
        row4.append(precision_recall(all_edges_score[m],k=0.75))
    print(row)
    writer.writerow(row)
    print(row2)
    writer.writerow(row2)
    print(row3)
    writer.writerow(row3)
    print(row4)
    writer.writerow(row4)


df = np.genfromtxt(toponame, delimiter=',',skip_header=1)

for m in range(2,2+len(methods)):
    print(m,df[2:,m].max())
    df[2:,m] = df[2:,m] / df[2:,m].max()
x=df[:,2:-1]
y=df[:,-1]
y



def reportClassifier(name,y_test,y_pred,scores=None):
    print("\n\t\t"+name)
    if(scores is not None):
        print("\nCross Validation Scores")
        print(scores)
    print("\nAccuracy: ",accuracy_score(y_test,y_pred))
    print("\nClassification Report")
    print(classification_report(y_test,y_pred))    
    print("\nConfusion Matrix")
    print(confusion_matrix(y_test,y_pred))
    try:
        print("\nROC AUC score")
        print(roc_auc_score(y_test, y_pred))
    except ValueError:
        print("Warning: Only one class in this fold")

def mean(x):
    return sum(x)/float(len(x))
 
def stddev(x):
    if(len(x)<2):
        return -1;
    avg = mean(x)
    variance = sum([pow(x-avg,2) for x in x])/float(len(x)-1)
    return math.sqrt(variance)

def getFeatureProbability(x, mean, stddev):
    if(stddev==0):# spike in gaussian
        return 1 if x==mean else 0
    exp = math.exp(-1*( math.pow(x-mean,2) / (2*math.pow(stddev,2))))
    return (1 / (math.sqrt(2*math.pi) * stddev)) * exp

def splitByLabel(x,y):
    yes,no=[],[]
    for i in range(len(y)):
        if(y[i]):
            yes.append(x[i,:])
        else:
            no.append(x[i,:])
    return yes,no

def getMeanStds(x_train):
    mean_stds=[]
    for i in range(len(x_train[0])): # 5 times
        mean_stds.append((mean(x_train[i]),stddev(x_train[i])))
    return mean_stds


def naive_bayes_gaussian(x_train,y_train):    
    h,w=x_train.shape
    labelYes,labelNo= splitByLabel(x_train,y_train)
    # Train part:
    yes_mean_stds=getMeanStds(labelYes)
    no_mean_stds=getMeanStds(labelNo)
    return yes_mean_stds,no_mean_stds

def predict_naive_bayes(x_test,yms,nms):    
    h,w=x_test.shape
    y_pred=[]
    #     p(Ci)
    p=len(yms)/(len(yms)+len(nms))
    # test part
    for i in range(h):
        x=x_test[i]
        p_yes,p_no=p,1-p
        for j in range(w): # 5 times
            m,s=yms[j]
            nm,ns=nms[j]
            p_yes *= getFeatureProbability(x[j],m,s)
            p_no *= getFeatureProbability(x[j],nm,ns)
        if(p_yes>=p_no):
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred


kf=KFold(n_splits=5,shuffle=True)
for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    yes_mean_stds,no_mean_stds = naive_bayes_gaussian(x_train,y_train)
    y_pred = predict_naive_bayes(x_test,yes_mean_stds,no_mean_stds)
    reportClassifier("GaussianUD",y_test,y_pred)

    b=GaussianNB()
    b.fit(x_train,y_train)
    y_pred=b.predict(x_test)
    reportClassifier("GaussianNB",y_test,y_pred)
        

svclassifier = SVC(kernel='linear')  
svclassifier.fit(x_train, y_train)
scores = cross_val_score(svclassifier,x, y, cv=5)
y_pred = svclassifier.predict(x_test)
reportClassifier("SVM",y_test,y_pred,scores)


clf = tree.DecisionTreeClassifier()
scores = cross_val_score(clf,x, y, cv=5)
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
reportClassifier("Decision Tree",y_test,y_pred,scores)


newdf = pd.DataFrame(x, columns=methods)
newdf = newdf.drop(['CN', 'AA'], axis =1)

### PCA 
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pc = pca.fit_transform(newdf)
principalDf = pd.DataFrame(data = pc, columns = ['pc_1', 'pc_2'])

## SVD
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(2)
sv = svd.fit_transform(newdf)
singulardf = pd.DataFrame(data = sv, columns = ['sv_1', 'sv_2'])
newdf['sv_1'] = singulardf['sv_1']
newdf['sv_2'] = singulardf['sv_2']

newdf = np.array(newdf)

kf=KFold(n_splits=5,shuffle=True)
# for i in range(0,5):
for train_index, test_index in kf.split(newdf):
    x_train, x_test = newdf[train_index], newdf[test_index]
    y_train, y_test = y[train_index], y[test_index]
    yes_mean_stds,no_mean_stds = naive_bayes_gaussian(x_train,y_train)
    y_pred = predict_naive_bayes(x_test,yes_mean_stds,no_mean_stds)
    #reportClassifier("GaussianUD",y_test,y_pred)

    #for trial-
    b=GaussianNB()
    b.fit(x_train,y_train)
    y_pred=b.predict(x_test)
    reportClassifier("GaussianNB",y_test,y_pred)

clf = tree.DecisionTreeClassifier()
scores = cross_val_score(clf,newdf, y, cv=5)
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
reportClassifier("Decision Tree",y_test,y_pred,scores)

svclassifier = SVC(kernel='linear')  
svclassifier.fit(x_train, y_train)
scores = cross_val_score(svclassifier,newdf, y, cv=5)
y_pred = svclassifier.predict(x_test)
reportClassifier("SVM",y_test,y_pred,scores)

import xgboost as xgb
dtrain = xgb.DMatrix(newdf, label = y)

params = {"objective" : "binary:logistic"}
model = xgb.cv(params, dtrain, num_boost_round = 40) 
model.loc[10:, ['test-error-mean', 'train-error-mean']].plot()
model

model = xgb.XGBClassifier(objective = 'binary:logistic')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
reportClassifier("XGB",y_test,y_pred,scores)