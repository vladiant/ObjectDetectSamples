# https://github.com/1297rohit/RCNN

import os

import pandas as pd

annot = "./Airplanes_Annotations"
path = "./Images"

with open("airplanes.info", "w") as f:
    for e1, i in enumerate(os.listdir(annot)):
        try:
            if i.startswith("airplane"):
                # Image
                filename = i.split(".")[0] + ".jpg"
                image_path = os.path.join(path, filename)

                counter = 0

                # Object ROI
                df = pd.read_csv(os.path.join(annot, i))
                gtvalues = str()
                for row in df.iterrows():
                    x1 = int(row[1][0].split(" ")[0])
                    y1 = int(row[1][0].split(" ")[1])
                    x2 = int(row[1][0].split(" ")[2])
                    y2 = int(row[1][0].split(" ")[3])

                    # Errors in size
                    if x2 > 255:
                        x2 = 255
                    if y2 > 255:
                        y2 = 255

                    gtvalues += f"{x1} {y1} {x2-x1} {y2-y1} "
                    counter += 1
                f.write(f"{image_path} {counter} {gtvalues}\n")

        except Exception as e:
            print(e)
            print(f"error in {filename}")
