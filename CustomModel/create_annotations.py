import os

import pandas as pd

import xml.etree.ElementTree as ET

import cv2 as cv

annot = "./Airplanes_Annotations"
path = "./Images"


for i in os.listdir(annot):
    try:
        if i.startswith("airplane"):
            # Image
            filename = i.split(".")[0] + ".jpg"
            filename_xml = i.split(".")[0] + ".xml"
            image_path = os.path.join(path, filename)
            xml_path = os.path.join(path, filename_xml)

            image = cv.imread(image_path)

            # Annotation data
            data = ET.Element("annotation")
            data.set("verified", "yes")
            folder = ET.SubElement(data, "folder")
            folder.text = "train"
            file_name = ET.SubElement(data, "filename")
            file_name.text = str(filename)
            file_path = ET.SubElement(data, "path")
            file_path.text = os.path.join("train", filename)
            source = ET.SubElement(data, "source")
            database = ET.SubElement(source, "database")
            database.text = "Unknown"

            image_size = ET.SubElement(data, "size")
            width = ET.SubElement(image_size, "width")
            width.text = str(image.shape[1])
            height = ET.SubElement(image_size, "height")
            height.text = str(image.shape[0])
            depth = ET.SubElement(image_size, "depth")
            if len(image.shape) > 2:
                depth.text = str(image.shape[2])
            else:
                depth.text = "1"

            segmented = ET.SubElement(data, "segmented")
            segmented.text = "0"

            # Object ROI
            df = pd.read_csv(os.path.join(annot, i))

            for row in df.iterrows():
                x1 = int(row[1][0].split(" ")[0])
                y1 = int(row[1][0].split(" ")[1])
                x2 = int(row[1][0].split(" ")[2])
                y2 = int(row[1][0].split(" ")[3])

                # Errors in size
                x2 = min(x2, 255)
                y2 = min(y2, 255)

                object = ET.SubElement(data, "object")
                name = ET.SubElement(object, "name")
                name.text = "airplane"
                pose = ET.SubElement(object, "pose")
                pose.text = "Unspecified"
                truncated = ET.SubElement(object, "truncated")
                truncated.text = "0"
                difficult = ET.SubElement(object, "difficult")
                difficult.text = "0"
                bndbox = ET.SubElement(object, "bndbox")
                xmin = ET.SubElement(bndbox, "xmin")
                xmin.text = str(x1)
                ymin = ET.SubElement(bndbox, "ymin")
                ymin.text = str(y1)
                xmax = ET.SubElement(bndbox, "xmax")
                xmax.text = str(x2)
                ymax = ET.SubElement(bndbox, "ymax")
                ymax.text = str(y2)

            with open(xml_path, "wb") as f:
                f.write(ET.tostring(data))
            print(f"{image_path} -> {xml_path}")

    except Exception as e:
        print(e)
        print(f"error in {filename}")
