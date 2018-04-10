"""
processes the images and returns a numpy array with the most significant features
"python.linting.pylintArgs": ["--extension-pkg-whitelist=numpy,cv2,PIL"]
"""

import sys

import cv2 as cv
#import numpy as np
#from PIL import Image


class g_obj(list):

    def __init__(self, clr):
        """
        stores the coordinates for the object on screen and stores a bounding box
        """
        super().__init__()
        self.left = sys.maxsize
        self.top = sys.maxsize
        self.right = 0
        self.bottom = 0
        self.color = clr

    def append(self, z):
        super().append(z)
        if z[0] < self.left:
            self.left = z[0]
        if z[0] > self.right:
            self.right = z[0]

        if z[1] < self.top:
            self.top = z[1]
        if z[1] > self.bottom:
            self.bottom = z[1]

    def check_lst(self, z):
        """
        checks a list of coordinates against the bounding box
        """
        for i in z:
            if self.check(i):
                return True
        return False

    def check(self, z):
        """
        checks a coordinate against the bounding box of the object
        """
        if z[0] >= self.left and z[0] <= self.right:
            if z[1] >= self.top and z[1] <= self.bottom:
                return True
        return False

    def merge(self, o):
        self += o

        if o.left < self.left:
            self.left = o.left

        if o.top < self.top:
            self.top = o.top

        if o.right > self.right:
            self.right = o.right

        if o.bottom > self.bottom:
            self.bottom = o.bottom


def process_image(path):
    """
    for now it converts the image to a numpy array
    """
    im = cv.imread(path)
    gr = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    #gr = cv.GaussianBlur(gr, (0,0), 2, 2)

    #try to find the background color
    #usually the color with the highest amount
    cr, cs = find_colors(gr)
    bc = sorted(cr, key=cr.get, reverse=True) #bc[0] is the highest amount of color

    #build the list of actually usable positions
    sc = []
    for k, i in cs.items():
        if k != bc[0]:
            sc += i

    #going over the image and finding object that are not background
    sc0 = (sc[0][0], sc[0][1])
    ob = [g_obj(gr[sc0[0]][sc0[1]])]
    ob[0].append(sc0)
    ob_map = {sc0 : 0} #maps positions to objects more space needed but less time

    def _m2o(pl):
        ts = set()
        for p in pl:
            if p in ob_map:
                ts.add(ob_map[p])
        for t in ts:
            yield t

    for x, y in sc[1:]:
        match = False
        for i in _m2o([(x-1, y-1), (x-1, y), (x, y-1), (x+1, y-1)]): #refine obj detection using g_obj merge
            if gr[x][y] == ob[i].color:
                ob[i].append((x, y))
                ob_map[(x, y)] = i
                match = True
                break
        if not match:
            ob.append(g_obj(gr[x][y]))
            ob[-1].append((x, y))
            ob_map[(x, y)] = len(ob)-1

    cv.imshow('grey', gr)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return ob

def find_colors(ar):
    """
    finds the different colors in the image
    returns a dict of colors with the amounts
    and returns a dict with all the coords for the colors
    """
    a = dict()
    b = dict()

    for row in range(ar.shape[0]):
        for col in range(ar.shape[1]):
            ac = ar[row][col]
            if ac in a:
                a[ac] += 1
                b[ac].append((row, col))
            else:
                a[ac] = 1
                b[ac] = [(row, col)]
    return a, b

process_image('./screenshots/frame20')
process_image('./screenshots/frame626')
