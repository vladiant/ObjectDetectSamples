# https://github.com/bikz05/bag-of-words

import cv2
import numpy as np
import os
import joblib

images_path = "./Images"
image = "Planes11.jpg"

# Set the random seed
np.random.seed(42)
cv2.setRNGSeed(42)

print("Loading model")

# Load the classifier
clf = cv2.ml.SVM_load("bof_svm.xml")

# Load the class names, scaler, number of clusters and labels
classes_names, means, stddevs, k, voc = joblib.load("bof.pkl")

# Intialize segmentation
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# Create feature extraction and keypoint detector objects
fea_det = cv2.BRISK_create()
des_ext = cv2.BRISK_create()

# Read image
image_path = os.path.join(images_path, image)
print(f"Read image {image_path}")
image = cv2.imread(image_path)
imread = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
imout = image.copy()

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
    timage = image[y: y + h, x: x + w]

    # Skip small segments
    if w < 6 or h < 6:
        continue

    kpts = fea_det.detect(timage)
    kpts, des = des_ext.compute(timage, kpts)

    # Skip images without features
    if des is None:
        continue

    descriptors = des.astype(np.float32)

    # Calculate the histogram of features
    test_features = np.zeros((1, k), np.float32)

    centers_indices = np.array([i for i in range(k)])
    knn = cv2.ml.KNearest_create()
    knn.train(voc, cv2.ml.ROW_SAMPLE, centers_indices)

    _, neigh, _, _ = knn.findNearest(descriptors, 1)
    for wt in neigh:
        test_features[0][int(wt)] += 1

    # Scale the features
    test_features = (test_features - means) / stddevs

    # Perform the predictions
    _, predictions = clf.predict(test_features.astype(np.float32))
    predictions = [
        classes_names[int(i)] for i in predictions.reshape(-1, 1).astype(np.int)
    ]

    # Draw segments with airplanes
    if int(predictions[0]) == 1:
        plane_segments.append((x, y, w, h))
        cv2.rectangle(imout, (x, y), (x + w, y + h), (0, 255, 0), 1)


cv2.imshow("Plane segments", imout)
cv2.waitKey(0)
