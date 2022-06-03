import numpy as np
import matplotlib.pyplot as plt
import pydicom
from sklearn.cluster import KMeans
import matplotlib.cm as cm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from sklearn.metrics import jaccard_score as jaccard

def kmeans_segmentation(im,nc):
    im[im > 3000] = 0 
    cjet = cm.viridis(range(256))
    im = im/np.max(im)
    im = np.array(im*255, dtype=np.uint8)
    im = cjet[im]
    dataK = np.zeros([im.shape[0]*im.shape[1], 3], dtype=np.double)
    dataK[:, 0] = np.reshape(im[:, :, 0], im.shape[0] * im.shape[1])
    dataK[:, 1] = np.reshape(im[:, :, 1], im.shape[0] * im.shape[1])
    dataK[:, 2] = np.reshape(im[:, :, 2], im.shape[0] * im.shape[1])
    
    kmeans_res = KMeans(n_clusters=nc, init='k-means++', random_state=0).fit(dataK / dataK.max())
    labels = kmeans_res.predict(dataK)
    
    imRes1 = np.reshape(labels, [im.shape[0], im.shape[1]])
    
    return imRes1


im = pydicom.dcmread('1924.dcm').pixel_array # Load CT image
nc=6 # Set number of clusters
im_segmented=kmeans_segmentation(im,nc)
plt.figure()
plt.imshow(im_segmented, cmap='nipy_spectral')
plt.colorbar()

# Comparation to the ground truth

im_gtm = plt.imread('liver_GT_024.png') # Get the ground truth mask

index = 0 # Set the index that gets the liver

print('SSIM: ')
print(ssim(np.float32(im_segmented==index), im_gtm)) # Print the SSIM value of the 2 images
print('MSE: ')
print(mse(np.float32(im_segmented==index), im_gtm)) # Print the MSE value of the 2 images
print('Jaccard Index: ')
print(jaccard(np.int8(im_gtm.flatten()),im_segmented.flatten()==index)) # Print the Jaccard index of the 2 images
