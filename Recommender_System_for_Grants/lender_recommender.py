import pandas as pd
import numpy as np
import pickle


def cosine_similarity(vec1, vec2):
    return (np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


# input Recipient amount and sector(s)
inp = np.empty([18])
fields = list(pd.read_csv("kiva_loans.csv//recep_data.csv").columns)
for i in fields:
    inp[fields.index(i)] = input(i+': ')

# # normalize grant amount
# df_norm = pd.read_csv("kiva_loans.csv//kiva_loans.csv")
# df_norm = df_norm[df_norm['country_code'] == 'IN']
# inp[0] = (inp[0] - df_norm['loan_amount'].min())/(df_norm['loan_amount'].max() - df_norm['loan_amount'].min())

# load classifier and classify input
clf = pickle.load(open("kiva_loans.csv//kmeans_recep.p", "rb"))
cluster = clf.predict(inp.reshape(1, -1))
print(cluster)

# load the cluster distances table
cluster_sim_tab = np.load(open("kiva_loans.csv//cluster_sim_tab.npy", "rb"))

# calculate cluster most similar to input's cluster
print(cluster_sim_tab[cluster, :])
nearest_cluster = np.where(cluster_sim_tab[cluster, :] == np.amax(cluster_sim_tab[cluster, :]))
print(nearest_cluster[0])

# read grant-writer data
df = pd.read_csv("kiva_loans.csv//lender_data.csv")
labels = pd.read_csv("kiva_loans.csv//lender_labels.csv")
df['labels'] = labels

# calculate similarities from input of all points within nearest cluster
df = df[df['labels'] == nearest_cluster[0][0]].loc[:, df.columns != 'labels']
df['sims'] = cosine_similarity(df[df.columns], inp)

# sort points in descending order of similarity with input
df.sort_values(by='sims', ascending=False).loc[:, df.columns != 'sims'].iloc[:20]
