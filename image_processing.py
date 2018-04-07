"""
processes the images and returns a numpy array with the most significant features
"python.linting.pylintArgs": ["--extension-pkg-whitelist=numpy,cv2,PIL"]
"""

<<<<<<< Updated upstream
import cv2 as cv
import numpy as np
import PIL
=======
import sys
>>>>>>> Stashed changes

import cv2 as cv
import numpy as np
from PIL import Image

def process_image(im):
    """
    for now it converts the image to a numpy array
    """
<<<<<<< Updated upstream
    im = cv.imread(path)
    gr = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    cr = find_colors(gr)
    cs = sorted(cr.keys(), reverse=True)

    print(cr)
    grr = reduce_color_range(gr, cs[0], cs[1])

    print(find_colors(grr))

    cv.imshow('grey', grr)
    cv.waitKey(0)
    cv.destroyAllWindows()

def reduce_color_range(ar, rpc, rdc):
    c = np.copy(ar)

    for row in range(len(c)):
        for col in range(c[row]):
            if c[row][col] == rpc:
                c[row][col] = rdc

    return c

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
=======

    img = Image.fromarray(im, 'RGB')
    img.convert('L')
    img.show()

process_image(cv.imread('./screenshots/frame0'))
>>>>>>> Stashed changes
