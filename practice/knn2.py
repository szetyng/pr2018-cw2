# attempt to remove images from the same camera
# using knn from sklearn - very slow performance

from scipy.io import loadmat
import numpy as np
from sklearn.neighbors import NearestNeighbors 
import time

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
gallery_cam = []
for i in gallery_idx:
    gallery_data.append(features[i])
    gallery_labels.append(labels[i])
    gallery_cam.append(camId[i])
gallery_data = np.array(gallery_data)
gallery_labels = np.array(gallery_labels)
gallery_cam = np.array(gallery_cam)

query_data = []
query_labels = []

for i in query_idx:
    query_data.append(features[i])
    query_labels.append(labels[i])
query_data = np.array(query_data)
query_labels = np.array(query_labels)

nn2_idx = []
start1 = time.time()
for j,query in enumerate(query_data):
    label = query_labels[j]
    cam = camId[query_idx[j]]

    gallery_idx_same = [a for a in gallery_idx if (labels[a]==label) and (camId[a] == cam)]
    gallery_idx_f = [b for b in gallery_idx if b not in gallery_idx_same]
    gallery_data = []
    for i in gallery_idx_f:
        gallery_data.append(features[i])
    gallery_data = np.array(gallery_data)

    knn2 = NearestNeighbors(n_neighbors=1)
    knn2.fit(gallery_data)
    nn2_idx.append(knn2.kneighbors(np.reshape(query, (1,2048)), return_distance=False))
    print(j)
end1 = time.time()
print(end1 - start1, 's')

nn2_idx = np.array(nn2_idx)
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