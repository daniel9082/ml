#Evaluate the performance of a hierarchical clustering algorithm on a dataset using different
evaluation metrics completeness score
# sillhuotte score
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score,completeness_score
print("Daniel MscIT 011")
data = np.array([[1,1],[5,5],[8,8],[1,0],[5,4],[8,1]])
linkage = 'ward'
model = AgglomerativeClustering(n_clusters=3,linkage=linkage)
model.fit(data)
cluster_labels = model.labels_
s = silhouette_score(data,cluster_labels)
print('Sillhuotte score:-',s)
g = None
if linkage== 'ward' and g is not None:
c = completeness_score(g,cluster_labels)
print(c)
else:
print('Not applicable')
