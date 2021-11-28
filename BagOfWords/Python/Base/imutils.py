#  https://github.com/bikz05/bag-of-words

import cv2, os
import matplotlib.pyplot as plt


def imlist(path):
    """
    The function imlist returns all the names of the files in
    the directory path supplied as argument to the function.
    """
    return [os.path.join(path, f) for f in os.listdir(path)]
