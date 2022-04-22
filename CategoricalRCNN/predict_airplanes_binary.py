# https://www.tensorflow.org/tutorials/images/cnn

from os import path

import numpy as np
import cv2 as cv

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)

from matplotlib import pyplot
from matplotlib.image import imread

# Source: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # Ff there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # If the bounding box is defined by integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + x1
    y2 = boxes[:, 3] + y1

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # Keep looping while some indexes still remain
    # in the indexes list
    while len(idxs) > 0:
        # Grab the last index in the indexes list and
        # add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x,y) coordinates for the start of
        # the bounding box and the smallest (x,y) coordinates
        # for the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.maximum(x2[i], x2[idxs[:last]])
        yy2 = np.maximum(y2[i], y2[idxs[:last]])

        # Compute the width and the height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that
        # have overlap more than threshold
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )

    # Return only the bounding boxes that were picked
    # using the integer data type
    return boxes[pick].astype("int")


print("Loading model")

model = tf.keras.models.load_model("airplanes_binary.h5")

print(model.summary())

# Intialize segmentation
cv.setUseOptimized(True)
ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()

images_path = "./Images"
image = "Planes11.jpg"

# Read image
image_path = path.join(images_path, image)
print(f"Read image {image_path}")
image = cv.imread(image_path)
imread = cv.cvtColor(image, cv.COLOR_BGR2RGB)
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
    timage = image[y : y + h, x : x + w]
    resized = cv.resize(timage, (32, 32), interpolation=cv.INTER_CUBIC)

    # predict
    resized = img_to_array(resized)
    resized = np.expand_dims(resized, axis=0)
    predictions = model.predict(resized)
    score = tf.nn.softmax(predictions[0])

    # Draw segments with airplanes
    if int(predictions[0]) == 1:
        plane_segments.append((x, y, w, h))
        cv.rectangle(imout, (x, y), (x + w, y + h), (0, 255, 0), 1)


cv.imshow("Plane segments", imout)
cv.waitKey(0)

# Check segments
print("Filter nonmax plane segments")
filtered_segments = non_max_suppression_fast(np.array(plane_segments), 0.7)
imout = image.copy()
for result in filtered_segments:
    x, y, w, h = result
    cv.rectangle(imout, (x, y), (x + w, y + h), (0, 255, 0), 1)

cv.imshow("Filtered Plane segments", imout)
cv.waitKey(0)
