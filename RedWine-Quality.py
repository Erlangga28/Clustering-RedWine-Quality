import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import sklearn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA

er = pd.read_csv('winequality-red.csv')
er.head()
print(er.head())

pH_Chart = sns.displot(er['pH'])

plt.figure(figsize = (13, 9))
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
visible = KElbowVisualizer(model, k=(1,10), timings = False)
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



