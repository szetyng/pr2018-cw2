from metric_learn import MMC_Supervised

from scipy.io import loadmat
import numpy as np
import time

import json
with open('PR_data/feature_data.json', 'r') as f:
    features = json.load(f)
features = np.array(features)

data = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')
#features = loadmat('PR_data/features.mat')['features']

camId = data['camId'].flatten()
filelist = data['filelist'].flatten() # list of arrays with only one value in it
gallery_idx = data['gallery_idx'].flatten()
labels = data['labels'].flatten()
query_idx = data['query_idx'].flatten()
train_idx = data['train_idx'].flatten()

# MATLAB arrays start with index = 1
query_idx = query_idx - 1
gallery_idx = gallery_idx - 1
train_idx = train_idx - 1

train_data = []
train_labels = []
for i in train_idx:
    train_data.append(features[i])
    train_labels.append(labels[i])

gallery_data = []
query_data = []
query_labels = []
for i in gallery_idx:
    gallery_data.append(features[i])
for i in query_idx:
    query_data.append(features[i])
    query_labels.append(labels[i])
query_labels = np.array(query_labels)

print('Done splitting data into training and testing')

mmc = MMC_Supervised(diagonal=True, num_constraints=1000)

start_fit = time.time()
mmc.fit(train_data, train_labels, random_state=np.random.RandomState(7))
end_fit = time.time()
print('Fitting training data took',end_fit-start_fit,'s')

start_transform = time.time()
gallery_data_r = mmc.transform(gallery_data)
query_data_r = mmc.transform(query_data)
end_transform = time.time()
print('Transforming test data took',end_transform-start_transform,'s')

# recreate feature matrix with transformed features
# preserve indices as defined 
features_r = np.zeros((features.shape[0], gallery_data_r.shape[1]))

for i in range(query_idx.shape[0]):
    features_r[query_idx[i]] = query_data_r[i]
for i in range(gallery_idx.shape[0]):
    features_r[gallery_idx[i]] = gallery_data_r[i]

#---------------------------------------------------------
# Evaluating the method
def knn_camspecific(k,features, labels, query_idx, gallery_idx, camId):
    knn_id = []
    #knn_dist = []
    for query_id in query_idx:
        label = labels[query_id]
        cam = camId[query_id]
        query = features[query_id]

        neighbourid_dist_pair = [(a, np.linalg.norm(np.array(query)-features[a])) for a in gallery_idx if not (labels[a]==label and camId[a]==cam)]
        neighbourid_dist_pair.sort(key = lambda x:x[1])

        kneighbour_id = [a[0] for a in neighbourid_dist_pair[:k]]
        #kdist = [a[1] for a in neighbourid_dist_pair[:k]]

        knn_id.append(kneighbour_id)
        #knn_dist.append(kdist)
    knn_id = np.array(knn_id)
    #knn_dist = np.array(knn_dist)
    return knn_id #, knn_dist

# Get k nearest neighbours
print('Starting knn')
start_knn = time.time()
nn_idx = knn_camspecific(10,features_r, labels, query_idx, gallery_idx, camId)
end_knn = time.time()
print('KNN took', end_knn-start_knn, 's')

nn_labels = np.empty(nn_idx.shape, dtype=np.uint16)
for i,neighbours in enumerate(nn_idx):
    for j,idx in enumerate(neighbours):
        nn_labels[i][j] = labels[idx]

# print accuracy @rankK for each k
klist = [1,5,10]
for k in klist:
    corr = []
    for i,neighbours in enumerate(nn_labels[:,:k]):
        if query_labels[i] in neighbours:
            corr.append(query_labels[i])
    acc = len(corr)/len(query_labels)
    print('acc'+str(k)+' = ', end='')
    print(acc*100)

#---------------------------------------------------------
# Analysing the method
M = mmc.metric()
print('M has shape', M.shape, 'and rank',np.linalg.matrix_rank(M))

L = mmc.transformer()
print('L has shape', L.shape, 'and rank',np.linalg.matrix_rank(L))




