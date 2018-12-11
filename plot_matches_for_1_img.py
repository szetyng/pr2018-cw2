import matplotlib.pyplot as plt

#%pylab inline
import matplotlib.image as mpimg
import cv2
from scipy.io import loadmat
import numpy as np
#from sklearn.decomposition import PCA, KernelPCA
import json
with open('PR_data/feature_data.json', 'r') as f:
    features = json.load(f)

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



# border widths; set them to 10
top, bottom, left, right = [10]*4
#define RGB colors
color_black = [0, 0, 0] 
color_red = [255,0,0]
color_gr = [0,150,0]

# for 1 query image: Returns the k nearest neighbours' indices and distances
def knn_camspecific1(k,features, labels, query_idx, gallery_idx, camId):
    knn_id = []
    knn_dist = []

    label = labels[query_idx]
    cam = camId[query_idx]
    query = features[query_idx]
    neighbourid_dist_pair = [(a, np.linalg.norm(np.array(query)-features[a])) for a in gallery_idx if not (labels[a]==label and camId[a]==cam)]
    neighbourid_dist_pair.sort(key = lambda x:x[1])

    kneighbour_id = [i[0] for i in neighbourid_dist_pair[:k]] #nn gallery_index
    kdist = [a[1] for a in neighbourid_dist_pair[:k]] #nn Euclidean distances

    knn_id.append(kneighbour_id)
    knn_dist.append(kdist)
    
    knn_id = np.array(knn_id)
    knn_dist = np.array(knn_dist)
    return knn_id, knn_dist

# SET test index randomly here:
query_test_ind = 543

# Get k nearest neighbours
nn_idx, nn_distances = knn_camspecific1(10,features, labels, query_test_ind, gallery_idx, camId)

nn_labels = np.empty(nn_idx.shape, dtype=np.uint16)
for i,neighbours in enumerate(nn_idx):
    for j,idx in enumerate(neighbours):
        nn_labels[i][j] = labels[idx]
#print(" nn_idx: ",nn_idx, "\n nn_labels[0]: ",nn_labels[0], "\n nn_distances:", nn_distances[0])


my_str = []
for i in nn_idx:
    my_str.append(filelist[i])
my_str = my_str[0]



#plot top 10 nearest neighbors for visualisation
collection=[]
for i, idx in enumerate(nn_labels[0]):
    img = mpimg.imread('PR_data/images_cuhk03/'+ str(my_str[i][0]))
    collection.append(img)
    if (nn_labels[0][i] == labels[query_test_ind]):
        img_with_border = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color_gr)
        collection[i] = img_with_border
        print("found: ",my_str[i][0])
    else: 
        img_wrong = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color_red)
        collection[i] = img_wrong
        print(i," not a match")


query_img = filelist[query_test_ind]
query = mpimg.imread('PR_data/images_cuhk03/'+ query_img[0])
q_img = cv2.copyMakeBorder(query, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color_black)
collection = [q_img]+collection

fig, axes = plt.subplots(1,11, figsize=(12,2.5))
eigenfaces = []
i=0
for ax,col in zip(axes.ravel(), collection[:11]):
    eigenfaces.append(col.real)
    ax.imshow(collection[i],aspect='auto')
    #ax.set_title('n '+str(i-1)), 
    ax.set_xticks([]), ax.set_yticks([])
    i+=1
plt.suptitle('Top 10 nn')
plt.show()

corr10 = 0
matched = []
for a, i in enumerate(nn_labels[0]):
    if (nn_labels[0][a] == labels[query_test_ind]):
        corr10+=1
        matched.append(a)
        print('a:',a,"|",corr10)
acc = corr10/len(nn_labels[0])
print("acc10",acc,matched)
