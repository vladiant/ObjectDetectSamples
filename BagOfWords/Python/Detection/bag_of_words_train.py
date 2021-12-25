# https://github.com/1297rohit/RCNN
# https://github.com/bikz05/bag-of-words
# https://docs.opencv.org/4.4.0/d1/d73/tutorial_introduction_to_svm.html
# https://docs.opencv.org/4.4.0/d1/d5c/tutorial_py_kmeans_opencv.html


import os
import cv2
import joblib

import pandas as pd

import numpy as np

annot = "./Airplanes_Annotations"
path = "./Images"

# Set the random seed
np.random.seed(42)
cv2.setRNGSeed(42)

# Create feature extraction and keypoint detector objects
fea_det = cv2.BRISK_create()
des_ext = cv2.BRISK_create()

# Initialize segmentation
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# List where all the descriptors are stored
des_list = []
image_classes = []

# Compute features
print("Compute features")
for e1, i in enumerate(os.listdir(annot)):
    try:
        if i.startswith("airplane"):
            # Image
            filename = i.split(".")[0] + ".jpg"
            print(f"Image number: {e1} , name: {filename}")
            image = cv2.imread(os.path.join(path, filename))

            mask = np.zeros((image.shape[1], image.shape[0]), dtype=np.uint8)

            # Object ROI
            df = pd.read_csv(os.path.join(annot, i))
            for row in df.iterrows():
                x1 = int(row[1][0].split(" ")[0])
                y1 = int(row[1][0].split(" ")[1])
                x2 = int(row[1][0].split(" ")[2])
                y2 = int(row[1][0].split(" ")[3])
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

            # Airplanes features
            kpts = fea_det.detect(image)
            kpts, des = des_ext.compute(image, kpts)

            des_list.append(des.astype(np.float32))
            image_classes.append(1)

            # Background features
            kpts = fea_det.detect(image, cv2.bitwise_not(mask))
            kpts, des = des_ext.compute(image, kpts)

            des_list.append(des.astype(np.float32))
            image_classes.append(0)

    except Exception as e:
        print(e)
        print(f"error processing {filename}")

# Stack all the descriptors vertically in a numpy array
print("Stack descriptors")
descriptors = des_list[0]
descriptor_labels = np.zeros((len(descriptors), 1), np.int)

for i, descriptor in enumerate(des_list[1:]):
    descriptor_label = np.zeros((len(descriptor), 1), np.int)
    descriptor_label.fill(i + 1)
    descriptor_labels = np.vstack((descriptor_labels, descriptor_label))
    descriptors = np.vstack((descriptors, descriptor))

# Perform k-means clustering
print("KMeans clustering")
k = 100
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 1e-6)
# KMEANS_RANDOM_CENTERS, KMEANS_PP_CENTERS
_, labels, voc = cv2.kmeans(descriptors, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

# Calculate the histogram of features
print("Calculate historgam of features")
im_features = np.zeros((len(des_list), k), np.float32)
for i, w in zip(descriptor_labels, labels):
    im_features[i[0]][w[0]] += 1

# Scaling the words
print("Normalize data")

means = []
stddevs = []
for i in range(im_features.shape[1]):
    mean, stdev = cv2.meanStdDev(im_features[:, i])
    means.append(mean[0][0])
    stddevs.append(stdev[0][0])
im_features = (im_features - means) / stddevs

# Train the Linear SVM
print("Train SVM")

clf = cv2.ml.SVM_create()
clf.setType(cv2.ml.SVM_C_SVC)
clf.setKernel(cv2.ml.SVM_LINEAR)
clf.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 50000, 1e-6))

train_data = cv2.ml.TrainData_create(
    im_features.astype(np.float32), cv2.ml.ROW_SAMPLE, np.array(image_classes)
)
clf.train(train_data)

score, data = clf.calcError(train_data, True)
print(f"Model accuracy: {1.0 - score/len(im_features)}")

# Save the SVM
clf.save("bof_svm.xml")

# Save the parameters
joblib.dump((["0", "1"], means, stddevs, k, voc), "bof.pkl", compress=3)
print("Done.")
