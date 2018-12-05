# attempt to remove images from the same camera

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

def knn_camspecific(k,features, labels, query_idx, gallery_idx, camId):
    knn_id = []
    #knn_dist = []
    for aa,query_id in enumerate(query_idx):
        label = labels[query_id]
        cam = camId[query_id]
        query = features[query_id]

        # gallery_data_f = []
        # gallery_idx_f = []

        neighbourid_dist_pair = []
        for a in gallery_idx:
            if not (labels[a]==label and camId[a]==cam):
                neighbourid_dist_pair.append([(a, np.linalg.norm(query-features[a]))])

        print(len(neighbourid_dist_pair))
        print(neighbourid_dist_pair[0])
        print(neighbourid_dist_pair[1])

        # for a in gallery_idx:
        #     if not (labels[a]==label and camId[a]==cam):
        #         gallery_data_f.append(features[a])
        #         gallery_idx_f.append(a)
        
        # neighbourid_dist_pair = [(gallery_idx_f[e], np.linalg.norm(query-gallery_data_f[e])) for e in range(len(gallery_idx_f))]
        neighbourid_dist_pair.sort(key = lambda x:x[1])

        kneighbour_id = [a[0] for a in neighbourid_dist_pair[:k]]
        #kdist = [a[1] for a in neighbourid_dist_pair[:k]]

        knn_id.append(kneighbour_id)
        #knn_dist.append(kdist)
        print(aa)
    knn_id = np.array(knn_id)
    #knn_dist = np.array(knn_dist)
    return knn_id#, knn_dist
    
query_labels = []
for i in query_idx:
    query_labels.append(labels[i])
query_labels = np.array(query_labels)

start1 = time.time()
nn2_idx = knn_camspecific(1,features, labels, query_idx[:5], gallery_idx, camId)
end1 = time.time()
print(end1 - start1, 's')

nn2_labels = np.empty(nn2_idx.shape, dtype=np.uint16)
for i,neighbours in enumerate(nn2_idx):
    for j,idx in enumerate(neighbours):
        nn2_labels[i][j] = labels[idx]

corr = []
for i,neighbours in enumerate(nn2_labels):
    if query_labels[i] in neighbours:
        corr.append(query_labels[i])
acc = len(corr)/len(query_labels)
print(acc)
