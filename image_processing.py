"""
processes the images and returns a numpy array with the most significant features
"""

import cv2
import numpy as np
import sys

def process_image(path):
    """
    for now it converts the image to a numpy array
    """
    im = cv2.imread(path)
    print(im)

process_image(sys.argv[1])