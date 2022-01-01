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


# Initialize segmentation
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

annot = "./Airplanes_Annotations"
path = "./Images"

iou_thresh = 0.99
samples_per_image = 30
out_width = 32
out_height = 32

sam_0_i = 0
sam_1_i = 0

# Extract images
for e1, i in enumerate(os.listdir(annot)):
    try:
        if i.startswith("airplane"):
            # Image
            filename = i.split(".")[0] + ".jpg"
            print(f"Image number: {e1} , name: {filename}")
            image = cv2.imread(os.path.join(path, filename))
            imout = image.copy()

            counter = 0

            # Object ROI
            df = pd.read_csv(os.path.join(annot, i))
            gtvalues = []
            for row in df.iterrows():
                x1 = int(row[1][0].split(" ")[0])
                y1 = int(row[1][0].split(" ")[1])
                x2 = int(row[1][0].split(" ")[2])
                y2 = int(row[1][0].split(" ")[3])
                gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
                timage = imout[y1:y2, x1:x2]

                # Set interpolation
                interpolation = cv2.INTER_AREA
                if timage.shape[0] > out_height or timage.shape[0] > out_width:
                    interpolation = cv2.INTER_CUBIC

                resized = cv2.resize(
                    timage, (out_height, out_width), interpolation=interpolation
                )

                object_image_path = os.path.join(path, "1", str(sam_1_i) + ".png")
                cv2.imwrite(object_image_path, resized)

                print(f"Object image number: {sam_1_i} , path: {object_image_path}")
                counter += 1
                sam_1_i += 1

            # Segmentation
            ss.setBaseImage(image)
            ss.switchToSelectiveSearchFast()
            ssresults = ss.process()

            falsecounter = 0
            samples_done = False
            positives_done = False
            negatives_done = False

            # Check segments
            for e, result in enumerate(ssresults):
                if e < 2000 and not samples_done:
                    # Check interlap with at least one segment
                    max_iou = 0
                    for gtval in gtvalues:
                        x, y, w, h = result
                        iou = get_iou(
                            gtval, {"x1": x, "x2": x + w, "y1": y, "y2": y + h}
                        )
                        if iou > max_iou:
                            max_iou = iou

                    if counter < samples_per_image:
                        if max_iou > iou_thresh:
                            timage = imout[y : y + h, x : x + w]

                            # Set interpolation
                            interpolation = cv2.INTER_AREA
                            if (
                                timage.shape[0] > out_height
                                or timage.shape[0] > out_width
                            ):
                                interpolation = cv2.INTER_CUBIC

                            resized = cv2.resize(
                                timage,
                                (out_height, out_width),
                                interpolation=interpolation,
                            )

                            object_overlap_image_path = os.path.join(
                                path, "1", str(sam_1_i) + ".png"
                            )
                            cv2.imwrite(object_overlap_image_path, resized)
                            print(
                                f"Object overlap image number: {sam_1_i} , path: {object_overlap_image_path}"
                            )
                            counter += 1
                            sam_1_i += 1
                    else:
                        positives_done = True

                    if falsecounter < len(gtvalues):
                        if max_iou <= 1 - iou_thresh:
                            timage = imout[y : y + h, x : x + w]

                            # Set interpolation
                            interpolation = cv2.INTER_AREA
                            if (
                                timage.shape[0] > out_height
                                or timage.shape[0] > out_width
                            ):
                                interpolation = cv2.INTER_CUBIC

                            resized = cv2.resize(
                                timage,
                                (out_height, out_width),
                                interpolation=interpolation,
                            )

                            backgroud_image_path = os.path.join(
                                path, "0", str(sam_0_i) + ".png"
                            )
                            cv2.imwrite(backgroud_image_path, resized)
                            print(
                                f"Background number: {sam_0_i} , path: {backgroud_image_path}"
                            )
                            falsecounter += 1
                            sam_0_i += 1
                    else:
                        negatives_done = True

                    if positives_done and negatives_done:
                        print("samples set completed")
                        samples_done = True
    except Exception as e:
        print(e)
        print(f"error processing {filename}")
