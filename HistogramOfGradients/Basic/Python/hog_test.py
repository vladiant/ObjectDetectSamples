# https://docs.opencv2.org/4.4.0/d1/d73/tutorial_introduction_to_svm.html

import cv2
import numpy as np
import os
import joblib

images_path = "./Images"
image = "Planes11.jpg"

SZ = 32  # Size
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


# Set the random seed
np.random.seed(42)
cv2.setRNGSeed(42)

print("Loading model")

# Load the classifier
clf = cv2.ml.SVM_load("svm_data.dat")

# Load the class names, scaler, number of clusters and labels
classes_names, means, stddevs, k, voc = joblib.load("bof.pkl")

# Intialize segmentation
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# Read image
image_path = os.path.join(images_path, image)
print(f"Read image {image_path}")
image = cv2.imread(image_path)
im_in = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
imout = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Segmentation
print("Performing image segmentation")

ss.setBaseImage(image)
# switchToSelectiveSearchFast, switchToSelectiveSearchQuality, switchToSingleStrategy
ss.switchToSingleStrategy()
ssresults = ss.process()

# Check segments
print("Extracting plane segments")

plane_segments = []
for result in ssresults:
    x, y, w, h = result
    timage = im_in[y : y + h, x : x + w]

    # Skip small segments
    if w < 6 or h < 6:
        continue

    deskewed = deskew(timage)
    hogdata = hog(deskewed)

    # Perform the predictions
    _, predictions = clf.predict(hogdata.reshape(1, -1).astype(np.float32))
    predictions = [
        classes_names[int(i)] for i in predictions.reshape(-1, 1).astype(np.int)
    ]

    # Draw segments with airplanes
    if int(predictions[0]) == 1:
        plane_segments.append((x, y, w, h))
        cv2.rectangle(imout, (x, y), (x + w, y + h), (0, 255, 0), 1)


cv2.imshow("Plane segments", imout)
cv2.waitKey(0)
