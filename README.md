Python file: liver_segment.py

Folder CT is the dataset. In this folder you can find a folder for each patient, and inside each of these 
folders, two subfolders can be seen:
- DICOM_anon: where all DICOM images can be found (all related to a sample number, example: i0023,0000b.dcm 
corresponds to number 23)
- Ground: where all Ground truth masks can be seen

1924.dcm contains the sample 24 from patient 19, and the ground truth mask is the .png that ends with
the number of the sample (24) 

2362.dcm contains the sample 62 from patient 23, and the ground truth mask is the .png that ends with
the number of the sample (62) 

In order to change files to segment/compare in the python script:
- Line 102 change the name of the DICOM image: the original is 1924.dcm 
- Line 111 change the name of the png image: the original is liver_GT_024.png

This repository also includes a python file for the segmentation with KMeans: kmeans.py
In order to change files to segment/compare in the python script:
- Line 29 change the name of the DICOM image: the original is 1924.dcm 
- Line 38 change the name of the png image: the original is liver_GT_024.png
