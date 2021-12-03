# https://github.com/bikz05/bag-of-words
# https://docs.opencv.org/4.4.0/d1/d73/tutorial_introduction_to_svm.html
# https://docs.opencv.org/4.4.0/d1/d5c/tutorial_py_kmeans_opencv.html

import argparse as ap
import cv2
import imutils
import numpy as np
import os

import joblib

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
args = vars(parser.parse_args())

# Get the training classes names and store them in a list
train_path = args["trainingSet"]
training_names = os.listdir(train_path)

# Set the random seed
np.random.seed(42)
cv2.setRNGSeed(42)

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
descriptor_labels = np.zeros((len(descriptors), 1), np.int)

for i, (image_path, descriptor) in enumerate(des_list[1:]):
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
joblib.dump((training_names, means, stddevs, k, voc), "bof.pkl", compress=3)
print("Done.")
