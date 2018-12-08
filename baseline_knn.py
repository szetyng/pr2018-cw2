# evaluating baseline approach by getting the k nearest neighbours of query data
# using our own knn algorithm to delete images from the same label and same camera

from scipy.io import loadmat
import numpy as np
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

query_idx = query_idx - 1
gallery_idx = gallery_idx - 1
train_idx = train_idx - 1

# SETTING K HERE
klist = [1,5,10,15,20]

# Returns the k nearest neighbours' indices and distances, for each query image
def knn_camspecific(k,features, labels, query_idx, gallery_idx, camId):
    knn_id = []
    knn_dist = []

    for aa,query_id in enumerate(query_idx):
        label = labels[query_id]
        cam = camId[query_id]
        query = features[query_id]

        neighbourid_dist_pair = [(a, np.linalg.norm(np.array(query)-features[a])) for a in gallery_idx if not (labels[a]==label and camId[a]==cam)]
        neighbourid_dist_pair.sort(key = lambda x:x[1])

        kneighbour_id = [a[0] for a in neighbourid_dist_pair[:k]]
        kdist = [a[1] for a in neighbourid_dist_pair[:k]]

        knn_id.append(kneighbour_id)
        knn_dist.append(kdist)
        print(aa)
    knn_id = np.array(knn_id)
    knn_dist = np.array(knn_dist)
    return knn_id, knn_dist
    
query_labels = []
for i in query_idx:
    query_labels.append(labels[i])
query_labels = np.array(query_labels)

# Get k nearest neighbours
kay = max(klist)
start1 = time.time()
nn_idx, nn_distances = knn_camspecific(kay,features, labels, query_idx, gallery_idx, camId)
end1 = time.time()
print(end1 - start1, 's')

nn_labels = np.empty(nn_idx.shape, dtype=np.uint16)
for i,neighbours in enumerate(nn_idx):
    for j,idx in enumerate(neighbours):
        nn_labels[i][j] = labels[idx]

# print accuracy for each k
for k in klist:
    corr = []
    for i,neighbours in enumerate(nn_labels[:,:k]):
        if query_labels[i] in neighbours:
            corr.append(query_labels[i])
    acc = len(corr)/len(query_labels)
    print('acc'+str(k)+' = ', end='')
    print(acc*100)


# for k = 10
corr10 = []
for i,neighbours in enumerate(nn_distances[:,:10]):
    if query_labels[i] in neighbours:
        corr10.append(query_labels[i])
acc10 = len(corr10)/len(query_labels)
print('acc10 =', acc10)


