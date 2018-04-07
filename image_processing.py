"""
processes the images and returns a numpy array with the most significant features
"python.linting.pylintArgs": ["--extension-pkg-whitelist=numpy,cv2,PIL"]
"""

import cv2 as cv
import numpy as np
from PIL import Image

def process_image(path):
    """
    for now it converts the image to a numpy array
    """
    im = cv.imread(path)
    gr = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    #gr = cv.GaussianBlur(gr, (0,0), 2, 2)

    #try to find the background color
    #usually the color with the highest amount
    cr = find_colors(gr)
    bc = sorted(cr.keys())
    print(bc)

    #going over the image and finding object that are not background
    #ob = {0 : (bc[0], [])}
    ob = {}

    #ed = cv.Canny(gr, bc[1], bc[-1], 3)

    for row in range(gr.shape[0]):
        for col in range(gr.shape[1]):
            if gr[row][col] != bc[0]:
                check_surroundings(ob, row, col, gr[row][col])
            #else:
            #    ob[0][1].append((row, col))

    print(ob)
    #for_del = []

    #for key, items in ob:
        #for pos in items[1]:


    cv.imshow('grey', gr)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return ob

def check_surroundings(dic, x, y, clr):
    if not dic:
        dic[0] = (clr,[(x, y)])
    else:
        for key, items in dic.items():
            if clr == items[0] and ((x-1, y-1) in items[1] or (x-1, y) in items[1] or (x, y-1) in items[1]):
                dic[key][1].append((x, y))
                return

        dic[max(dic.keys())+1] = (clr,[(x, y)])

def find_colors(ar):
    c = dict()

    for row in ar:
        for el in row:
            if el in c:
                c[el] += 1
            else:
                c[el] = 1
    return c

process_image('./screenshots/frame500')
process_image('./screenshots/frame626')
