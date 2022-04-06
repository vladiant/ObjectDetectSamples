# https://towardsdatascience.com/step-by-step-r-cnn-implementation-from-scratch-in-python-e97101ccde55
# https://github.com/1297rohit/RCNN

import os
import cv2

import pandas as pd

import numpy as np


# Intersection over Union https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def get_iou(bb1, bb2):
    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["x1"] < bb2["x2"]

    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    assert iou >= 0.0
    assert iou <= 1.0

    return iou


# Set the random seed
np.random.seed(42)
cv2.setRNGSeed(42)

annot = "./Airplanes_Annotations"
path = "./Images"

iou_data = list()

# Initialize segmentation
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# Extract images
for e1, i in enumerate(os.listdir(annot)):
    try:
        if i.startswith("airplane"):
            # Image
            filename = i.split(".")[0] + ".jpg"
            # print(f"Image number: {e1} , name: {filename}")
            image_filename = os.path.join(path, filename)
            image = cv2.imread(image_filename)
            imout = image.copy()

            # Object ROI
            df = pd.read_csv(os.path.join(annot, i))
            gtvalues = []
            for row in df.iterrows():
                x1 = int(row[1][0].split(" ")[0])
                y1 = int(row[1][0].split(" ")[1])
                x2 = int(row[1][0].split(" ")[2])
                y2 = int(row[1][0].split(" ")[3])

                # Errors in size
                x2 = min(x2, 255)
                y2 = min(y2, 255)

                gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})

                # Image where object is
                timage = imout[y1:y2, x1:x2]
                iou_data.append((1.0, x1, x2, y1, y2, image_filename))

            # Segmentation
            ss.setBaseImage(image)
            ss.switchToSelectiveSearchFast()
            ssresults = ss.process()

            # Check segments
            for result in ssresults:
                # Check interlap with not more than one segment
                y1, y2, x1, x2 = 0.0, 0.0, 0.0, 0.0
                local_iou = 0.0
                positive_iou_count = 0
                for gtval in gtvalues:
                    x, y, w, h = result
                    iou = get_iou(gtval, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
                    # print(f"Image number: {e1} , name: {filename}, iou: {iou}")
                    if not np.isclose(iou, 0):
                        positive_iou_count += 1
                        local_iou = iou, x, x + w, y, y + h, image_filename

                if positive_iou_count == 1:
                    iou_data.append(local_iou)

    except Exception as e:
        print(e)
        print(f"error processing {filename}")

# Write to file
with open("iou_data.csv", "w") as output_file:
    output_file.write("IOU, x1, x2, y1, y2, filename\n")
    for elem in iou_data:
        output_file.write(f"{', '.join([str(x) for x in elem])}\n")


# Calculate histogram
hist, bin_edges = np.histogram([elem[0] for elem in iou_data], bins=100, density=False)
print(hist)
print(bin_edges)
