import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from skimage import measure
import cv2
import pydicom
from scipy.signal import medfilt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from sklearn.metrics import jaccard_score as jaccard

def liver_segment(im,eq):
    im[im > 3000] = 0
    im = np.array((im/np.max(im))*255, dtype=np.uint8)
    if eq:
        im=equalization(im)

    # Smoothing and quantization
    # CROPPING IMAGE, threshold=0.035
    im = im/np.max(im)  # to get values between 0-1
    im[im < 0.035] = 0
    # Morphological opening/closing, 5x5 kernel
    closed = closing(opening(im, 5),5)   
    quantized = np.floor(medfilt(closed, 7)*255/64)
    quantized = np.array(quantized, dtype=np.uint8)
    # Get the seeds and perform a region growing on the quantized image
    return region_grow(quantized,get_seeds(quantized))
    

def opening(image, kernelSize):
    kernel = np.ones((kernelSize,kernelSize), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opening


def closing(image, kernelSize):
    kernel = np.ones((kernelSize,kernelSize), np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closing

def equalization(data):
    hg = sp.ndimage.histogram(data, 0, 255, 256)
    chg = np.cumsum(hg)
    eqim = np.uint8((255*chg[data])/(data.shape[0]*data.shape[1]))
    return eqim

def get_seeds(quantized):
    seeds=np.zeros([30,3],int) #Create a list to store the seeds
    i=0
    for seg1 in np.delete(np.unique(quantized), np.where(np.unique(quantized) == 0)): # For each quantized value that is not 0 (background)
        im_seg=np.copy(quantized)==seg1
        im_seg = measure.label(im_seg) # Get the connected components for each segment
        for seg2 in np.delete(np.unique(im_seg), np.where(np.unique(im_seg) == 0)):
            if np.sum(im_seg==seg2)>=196: # If the segment is greater than 14x14
                ind=np.where(im_seg==seg2)
                xm=np.median(ind[0]) # Get the median indexes
                ym=np.median(ind[1])
                xyn=np.sum(np.abs(np.transpose(ind)-[xm,ym]),1).argmin() # Get the point of the segment that is closest to the median indexes
                [xs,ys]=[ind[0][xyn],ind[1][xyn]]
                seeds[i,:]=[seg1,xs,ys] # Add those indexes to the seed point list
                i+=1
    return np.delete(seeds,range(i,30),0)

def region_grow(im1,seeds):
    i=1 # This will be the value of each region and will be increased by 1 for each different seed point
    im_gs=np.zeros([im1.shape[0],im1.shape[1]]) # Create a new image for the region growing
    im1=im1.astype(int)
    for [l,x,y] in seeds: # For each seed
        Qm=im1[x,y] # Set the mean intensity of the region as the intensity of the seed point
        new_points=[[x,y]] # Create a list to put all the new points added to the region and add the seed point
        while new_points: # While there has been new points added to the region
            new_points1=new_points
            new_points=[]
            for [x,y] in new_points1: # For each new point
                if x < 511 and y < 511: # If any of his neighbors has a intensity that differs < 0.2 from the seed point intensity 
                                        # Add this point as a new point and paint it to the image with the value of the region
                    if abs(im1[x+1,y]-Qm)<0.2 and im_gs[x+1,y]==0:
                        im_gs[x+1,y]=i
                        new_points.append([x+1,y])
                    if abs(im1[x-1,y]-Qm)<0.2 and im_gs[x-1,y]==0:
                        im_gs[x-1,y]=i
                        new_points.append([x-1,y])
                    if abs(im1[x,y+1]-Qm)<0.2 and im_gs[x,y+1]==0:
                        im_gs[x,y+1]=i
                        new_points.append([x,y+1])
                    if abs(im1[x,y-1]-Qm)<0.2 and im_gs[x,y-1]==0:
                        im_gs[x,y-1]=i
                        new_points.append([x,y-1])
        i+=1 # Increase the value of the region
    return im_gs
 
# Image segmentation

im = pydicom.dcmread('1924.dcm').pixel_array # Get the CT image
im_segmented=liver_segment(im,1) # Segment the image

plt.figure()
plt.imshow(im_segmented, cmap='nipy_spectral')
plt.colorbar()

# Comparation to the ground truth

im_gtm = plt.imread('liver_GT_024.png') # Get the ground truth mask

index = 8 # Set the index that gets the liver

print('SSIM: ')
print(ssim(np.float32(im_segmented==index), im_gtm)) # Print the SSIM value of the 2 images
print('MSE: ')
print(mse(np.float32(im_segmented==index), im_gtm)) # Print the MSE value of the 2 images
print('Jaccard Index: ')
print(jaccard(np.int8(im_gtm.flatten()),im_segmented.flatten()==index)) # Print the Jaccard index of the 2 images


