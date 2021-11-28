# https://github.com/bikz05/bag-of-words

import argparse as ap
import cv2
import imutils
import numpy as np
import os
from sklearn.svm import LinearSVC
import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
args = vars(parser.parse_args())

# Set the random seed
np.random.seed(42)
cv2.setRNGSeed(42)

# Get the training classes names and store them in a list
train_path = args["trainingSet"]
training_names = os.listdir(train_path)

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
print("Load images")
image_paths = []
image_classes = []
class_id = 0
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imutils.imlist(dir)
    image_paths += class_path
    image_classes += [class_id] * len(class_path)
    class_id += 1

# Create feature extraction and keypoint detector objects
fea_det = cv2.BRISK_create()
des_ext = cv2.xfeatures2d.DAISY_create()

# List where all the descriptors are stored
des_list = []
updated_image_classes = []

# Compute features
print("Compute features")
for image_path, image_class in zip(image_paths, image_classes):
    im = cv2.imread(image_path)
    kpts = fea_det.detect(im)
    kpts, des = des_ext.compute(im, kpts)

    # Skip images without features
    if des is None:
        continue

    updated_image_classes.append(image_class)
    des_list.append((image_path, des.astype(np.float32)))

# Update image classes
image_classes = updated_image_classes

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]

for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

# Perform k-means clustering
print("KMeans clustering")
k = 100
voc, variance = kmeans(descriptors, k, 1)

# Calculate the histogram of features
print("Calculate historgam of features")
im_features = np.zeros((len(des_list), k), np.float32)
for i in range(len(des_list)):
    words, distance = vq(des_list[i][1], voc)
    for w in words:
        im_features[i][w] += 1

# Scaling the words
print("Normalize data")
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

# Train the Linear SVM
print("Train SVM")
clf = LinearSVC(max_iter=10000)
clf.fit(im_features, np.array(image_classes))
print(f"train score: {clf.score(im_features, np.array(image_classes))}")

# Save the SVM
joblib.dump((clf, training_names, stdSlr, k, voc), "bof.pkl", compress=3)
print("Done.")
