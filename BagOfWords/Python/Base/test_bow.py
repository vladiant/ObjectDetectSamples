# https://github.com/bikz05/bag-of-words

import argparse as ap
import cv2
import imutils
import numpy as np
import os
from sklearn.svm import LinearSVC
import joblib
from scipy.cluster.vq import *

# Set the random seed
np.random.seed(42)
cv2.setRNGSeed(42)

# Load the classifier, class names, scaler, number of clusters and vocabulary
clf, classes_names, stdSlr, k, voc = joblib.load("bof.pkl")

# Get the path of the testing set
parser = ap.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-t", "--testingSet", help="Path to testing Set")
group.add_argument("-i", "--image", help="Path to image")
parser.add_argument("-v", "--visualize", action="store_true")
args = vars(parser.parse_args())

# Get the path of the testing image(s) and store them in a list
print("Read image(s)")
image_paths = []
if args["testingSet"]:
    test_path = args["testingSet"]
    try:
        testing_names = os.listdir(test_path)
    except OSError:
        print(f"No such directory {test_path}\nCheck if the file exists")
        exit()
    for testing_name in testing_names:
        dir = os.path.join(test_path, testing_name)
        class_path = imutils.imlist(dir)
        image_paths += class_path
else:
    image_paths = [args["image"]]

# Create feature extraction and keypoint detector objects
fea_det = cv2.BRISK_create()
des_ext = cv2.xfeatures2d.DAISY_create()

print("Compute features")
# List where all the descriptors are stored
des_list = []

processed_image_paths = []
for image_path in image_paths:
    im = cv2.imread(image_path)
    if im is None:
        print(f"No such file {image_path}\nCheck if the file exists")
        exit()
    kpts = fea_det.detect(im)
    kpts, des = des_ext.compute(im, kpts)

    # Skip images without features
    if des is None:
        continue

    des_list.append((image_path, des.astype(np.float32)))
    processed_image_paths.append(image_path)

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

# Calculate the histogram of features
print("Calculate historgam of features")
test_features = np.zeros((len(des_list), k), np.float32)
for i in range(len(des_list)):
    words, distance = vq(des_list[i][1], voc)
    for w in words:
        test_features[i][w] += 1

# Scale the features
print("Normalize data")
test_features = stdSlr.transform(test_features)

# Perform the predictions
print("Predict classes")
predictions = [classes_names[i] for i in clf.predict(test_features)]

# Visualize the results, if "visualize" flag set to true by the user
if args["visualize"]:
    print("Show images with predictions")
    for image_path, prediction in zip(processed_image_paths, predictions):
        image = cv2.imread(image_path)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        pt = (0, 3 * image.shape[0] // 4)
        cv2.putText(
            image, prediction, pt, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, [0, 255, 0], 2
        )
        cv2.imshow("Image", image)
        print(f"Image: {image_path}, Predicted class: {prediction}")
        cv2.waitKey(3000)

print("Done.")
