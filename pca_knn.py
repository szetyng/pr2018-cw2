# evaluating baseline approach by getting the k nearest neighbours of query data
# using our own knn algorithm to delete images from the same label and same camera

from scipy.io import loadmat
import numpy as np
import time
from sklearn.decomposition import PCA
#from metric_learn import MMC_Supervised
import json
with open('PR_data/feature_data.json', 'r') as f:
    features = json.load(f)
features = np.array(features)

data = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')
camId = data['camId'].flatten()
filelist = data['filelist'].flatten() # list of arrays with only one value in it
gallery_idx = data['gallery_idx'].flatten()
labels = data['labels'].flatten()
query_idx = data['query_idx'].flatten()
train_idx = data['train_idx'].flatten()

#  Convert from Matlab indexing to Python indexing
gallery_idx = gallery_idx - 1
query_idx = query_idx - 1
train_idx = train_idx - 1

query_labels = []
query_data = []
for i in query_idx:
    query_labels.append(labels[i])
    query_data.append(features[i])
train_data = []
for i in train_idx:
    train_data.append(features[i])
gallery_data = []
for i in gallery_idx:
    gallery_data.append(features[i])

def knn_camspecific(k,features, labels, query_idx, gallery_idx, camId):
    knn_id = []
    #knn_dist = []
    for aa,query_id in enumerate(query_idx):
        label = labels[query_id]
        cam = camId[query_id]
        query = features[query_id]

        neighbourid_dist_pair = [(a, np.linalg.norm(np.array(query)-features[a])) for a in gallery_idx if not (labels[a]==label and camId[a]==cam)]
        neighbourid_dist_pair.sort(key = lambda x:x[1])

        kneighbour_id = [a[0] for a in neighbourid_dist_pair[:k]]
        #kdist = [a[1] for a in neighbourid_dist_pair[:k]]

        knn_id.append(kneighbour_id)
        #knn_dist.append(kdist)
        #print(aa)
    knn_id = np.array(knn_id)
    #knn_dist = np.array(knn_dist)
    return knn_id#, knn_dist
    

mpca = [2048, 1024, 512, 256]

for m in mpca:
    pca = PCA(n_components=m, random_state=7)
    start_fit = time.time()
    pca.fit(train_data)
    end_fit = time.time()
    print('Training PCA for mpca of', m, 'took', end_fit-start_fit, 's')
    
    start_trans = time.time()
    gallery_pca = pca.transform(gallery_data)
    query_pca = pca.transform(query_data)
    end_trans = time.time()
    print('Transforming test data took', end_trans-start_trans, 's')

    features_pca = np.zeros((features.shape[0], gallery_pca.shape[1]))
    for i in range(query_idx.shape[0]):
        features_pca[query_idx[i]] = query_pca[i]
    for i in range(gallery_idx.shape[0]):
        features_pca[gallery_idx[i]] = gallery_pca[i]

    start_knn = time.time()
    nn_idx = knn_camspecific(10, features_pca, labels, query_idx, gallery_idx, camId)
    end_knn = time.time()
    print('KNN took', end_knn-start_knn, 's')

    nn_labels = np.empty(nn_idx.shape, dtype=np.uint16)

    for i,neighbours in enumerate(nn_idx):
        for j,idx in enumerate(neighbours):
                nn_labels[i][j] = labels[idx]

    # print accuracy for each k
    klist = [1,5,10]
    acclist = []
    for k in klist:
        corr = []
        for i,neighbours in enumerate(nn_labels[:,:k]):
            if query_labels[i] in neighbours:
                corr.append(query_labels[i])
        acc = len(corr)/len(query_labels)
        print('acc'+str(k)+' = ', end='')
        print(acc*100)
        acclist.append(acc*100)

