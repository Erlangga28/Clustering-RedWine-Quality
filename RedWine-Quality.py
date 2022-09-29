import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import sklearn
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.cluster import AgglomerativeClustering

er = pd.read_csv('winequality-red.csv') #read imported data
er.head()
print(er.head()) #to show the data

pH_Chart = sns.displot(er['pH']) #making chart of 'pH' data

plt.figure(figsize = (13, 9))
#er = er.drop('alcohol', axis = 1)
er = er.drop('pH', axis = 1)
er.columns

wcss = []
for i in range (1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(er)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Methods Graphics')
plt.xlabel('Cluster')
plt.ylabel('WCSS')
plt.show()

model = KMeans()
visible = KElbowVisualizer(model, k=(1,11), timings = False)
visible.fit(er)
visible.show()

bca = PCA()
X = bca.fit_transform(er)
kmeans = KMeans(n_clusters=3)
label = kmeans.fit_predict(X)
graphics = np.unique(label)

for i in graphics:
    plt.scatter(X[label==i,0], X[label==i,1], label=i, s=20)
    
plt.legend()
plt.title('Wine After Drop pH')
plt.show()

#Making the Dendrogam Graf
plt.figure(figsize= (8, 4))
Coalition = linkage(er,method='ward')
dendrogram(Coalition)
plt.axhline(y=5, color='r', linestyle='--')
plt.show()

cl = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='average')
cl.fit_predict(er)
print(cl.fit_predict(er))

plt.figure(figsize= (8, 4))
Coalition = linkage(er,method='centroid')
dendrogram(Coalition)
plt.axhline(y=2.5, color='r', linestyle='--')
plt.show()

cl = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
cl.fit_predict(er)
print(cl.fit_predict(er))

plt.figure(figsize= (8, 4))
Coalition = linkage(er,method='centroid')
dendrogram(Coalition)
plt.axhline(y=1, color='r', linestyle='--')
plt.show()

cl = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='single')
cl.fit_predict(er)
print(cl.fit_predict(er))
#Erlangga Wahyu Utomo
#5025201118
