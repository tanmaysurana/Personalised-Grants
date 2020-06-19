import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle
import math

def cosine_similarity(vec1, vec2):
    return (np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

# read recipient data
recep = pd.read_csv("kiva_loans.csv//recep_data.csv")

# read lender data
lender = pd.read_csv("kiva_loans.csv//lender_data.csv")

# apply k-means to form clusters
# n_clusters becomes 10x when len(recep) becomes 10x
n_clusters = 10**int(math.log(len(recep))/math.log(1000))
kmeans_recep = KMeans(n_clusters = n_clusters).fit(np.array(recep))
pickle.dump(kmeans_recep, open("kiva_loans.csv//kmeans_recep.p", "wb"))
kmeans_lender = KMeans(n_clusters = n_clusters).fit(np.array(lender))
pickle.dump(kmeans_lender, open("kiva_loans.csv//kmeans_lender.p", "wb"))
recep['label'] = kmeans_recep.labels_
lender['label'] = kmeans_lender.labels_
pd.DataFrame(kmeans_recep.labels_, columns=['labels']).to_csv("kiva_loans.csv\\recep_labels.csv", index=False)
pd.DataFrame(kmeans_lender.labels_, columns=['labels']).to_csv("kiva_loans.csv\\lender_labels.csv", index=False)

# calculate similarities between all clusters and save them in a 2d numpy array
cluster_sim_tab = np.empty([n_clusters, n_clusters])
for i in range(n_clusters):
    for j in range(n_clusters):
        df1 = recep[recep['label'] == i].loc[:, recep.columns != 'label'].mean()
        print(df1.head())
        df2 = lender[lender['label'] == j].loc[:, lender.columns != 'label'].mean()
        print(df2.head())
        cluster_sim_tab[i, j] = cosine_similarity(np.array(df1), np.array(df2))
print(cluster_sim_tab)
np.save(open("kiva_loans.csv//cluster_sim_tab.npy", "wb"), cluster_sim_tab)
