# did not remove images from the same camera

from scipy.io import loadmat
import numpy as np
from sklearn.neighbors import NearestNeighbors 
# import json
# with open('PR_data/feature_data.json', 'r') as f:
#     features = json.load(f)

data = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')
features = loadmat('PR_data/features.mat')['features']

camId = data['camId'].flatten()
filelist = data['filelist'].flatten() # list of arrays with only one value in it
gallery_idx = data['gallery_idx'].flatten()
labels = data['labels'].flatten()
query_idx = data['query_idx'].flatten()
train_idx = data['train_idx'].flatten()

gallery_data = []
gallery_labels = []
for i in gallery_idx:
    gallery_data.append(features[i])
    gallery_labels.append(labels[i])
gallery_data = np.array(gallery_data)
gallery_labels = np.array(gallery_labels)

query_data = []
query_labels = []
for i in query_idx:
    query_data.append(features[i])
    query_labels.append(labels[i])
query_data = np.array(query_data)
query_labels = np.array(query_labels)

# knn1 = NearestNeighbors(n_neighbors=1)
# knn1.fit(gallery_data)
# nn1_idx = knn1.kneighbors(query_data, return_distance=False).flatten()

# nn1_labels = []
# for i in nn1_idx:
#     nn1_labels.append(gallery_labels[i])
# nn1_labels = np.array(nn1_labels)

# # for k = 1, label of neighbour is the prediction
# corr = []
# for i,ytilda in enumerate(nn1_labels):
#     if ytilda == query_labels[i]:
#         corr.append(ytilda)
# acc = len(corr)/len(query_labels)
# print(acc)

knn2 = NearestNeighbors(n_neighbors=2)
knn2.fit(gallery_data)
# find k nearest neighbours
nn2_idx = knn2.kneighbors(np.reshape(query_data[0], (1,2048)), return_distance=False) # array of arrays, inner array is length k

nn2_labels = np.empty(nn2_idx.shape, dtype=np.uint16)
for i,neighbours in enumerate(nn2_idx):
    for j,idx in enumerate(neighbours):
        nn2_labels[i][j] = gallery_labels[idx]

corr = []
for i,neighbours in enumerate(nn2_labels):
    if query_labels[i] in neighbours:
        corr.append(query_labels[i])
acc = len(corr)/len(query_labels)
print(acc)