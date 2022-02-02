# https://docs.opencv2.org/4.4.0/d1/d73/tutorial_introduction_to_svm.html

import argparse as ap
import cv2
import numpy as np
import os

import joblib

SZ = 20  # Size
bin_n = 16  # Number of bins
affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR


def imlist(path):
    """
    The function imlist returns all the names of the files in
    the directory path supplied as argument to the function.
    """
    return [os.path.join(path, f) for f in os.listdir(path)]


def deskew(img):
    m = cv2.moments(img)
    if abs(m["mu02"]) < 1e-2:
        return img.copy()
    skew = m["mu11"] / m["mu02"]
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
    return img


def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n * ang / (2 * np.pi))  # quantizing binvalues in (0...16)
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [
        np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)
    ]
    hist = np.hstack(hists)  # hist is a 64 bit vector
    return hist


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
    class_path = imlist(dir)
    image_paths += class_path
    image_classes += [class_id] * len(class_path)
    class_id += 1

# List where all the descriptors are stored
image_hog = []

# Compute features
print("Compute histogram of gradients")
for image_path, image_class in zip(image_paths, image_classes):
    im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    deskewed = deskew(im)
    hogdata = hog(deskewed)
    image_hog.append(hogdata.astype(np.float32))

# Prepare model
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)

# Prepare samples data
image_hog = np.float32(image_hog).reshape(-1, 64)
image_classes = np.array(image_classes)
train_data = cv2.ml.TrainData_create(image_hog, cv2.ml.ROW_SAMPLE, image_classes)

# Train/test split, shuffle
train_data.setTrainTestSplitRatio(0.8, True)

print("Train SVM")
svm.train(train_data)

train_score, _ = svm.calcError(train_data, False)
print(f"Train accuracy: {1.0 - train_score/len(image_classes)}")

test_score, _ = svm.calcError(train_data, True)
print(f"Test accuracy: {1.0 - test_score/len(image_classes)}")

print("Save trained model data as svm_data.dat")
svm.save("svm_data.dat")

print("Done.")
