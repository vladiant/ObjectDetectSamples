# https://pythonprogramming.net/haar-cascade-object-detection-python-opencv-tutorial/
from os import path

import numpy as np
import cv2

images_path = "Images"
image = "Planes11.jpg"

cascades_path = "data"
cascade = "cascade.xml"

image_path = path.join(images_path, image)
cascade_path = path.join(cascades_path, cascade)

object_cascade = cv2.CascadeClassifier(cascade_path)

img = cv2.imread(image_path, cv2.IMREAD_COLOR)

watches = object_cascade.detectMultiScale(img, 4, 50)

for (x, y, w, h) in watches:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

cv2.imshow("img", img)
cv2.waitKey(0)

cv2.destroyAllWindows()
