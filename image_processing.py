"""
processes the images and returns a numpy array with the most significant features
"python.linting.pylintArgs": ["--extension-pkg-whitelist=numpy,cv2,PIL"]
"""

import sys

import cv2 as cv
#import numpy as np
from PIL import Image


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
        """
        merges o into self
        """
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
    rs = cv.resize(im, (int(im.shape[1]/6), int(im.shape[0]/6)))
    cv.imwrite("6xdown.png", rs)
    gr = cv.cvtColor(rs, cv.COLOR_BGR2GRAY)
    cv.imwrite("gray.png", gr)

    #try to find the background color
    #usually the color with the highest amount
    cr, cs, dv, dl = pre_process(gr)
    bc = sorted(cr, key=cr.get, reverse=True) #bc[0] is the highest amount of color
    bkg = bc[0]

    #build the list of actually usable positions
    sc = []
    for k, i in cs.items():
        if k != bkg:
            sc += i

    #going over the image and finding object that are not background
    sc0 = (sc[0][0], sc[0][1])
    ob = {0: g_obj(gr[sc0[0]][sc0[1]])}
    ob[0].append(sc0)
    ob_map = {sc0 : 0} #maps positions to objects more space needed but less time

    def _m2o(clr, pl):
        """
        returns a list of unique objects around the pos
        """
        ts = set()
        for p in pl:
            if p in ob_map:
                if ob[ob_map[p]].color == clr:
                    ts.add(ob_map[p])
        return list(ts)

    for x, y in sc[1:]:
        ls = _m2o(gr[x][y], [(x-1, y-1), (x-1, y), (x, y-1), (x+1, y-1)]) #refine obj detection using g_obj merge
        g = g_obj(gr[x][y])
        g.append((x, y))
        ind = max(ob.keys())+1

        if len(ls) is 1:
            ob[ls[0]].append((x, y))
            ob_map[(x, y)] = ls[0]
            continue

        elif len(ls) > 1:
            for o in ls:
                g.merge(ob[o])
                for obp in ob[o]:
                    ob_map[obp] = ind
                del ob[o]

        ob[ind] = g
        ob_map[(x, y)] = ind

    #for k, v in ob.items():
    #    cv.rectangle(rs, (v.top, v.left), (v.bottom, v.right), (0, 255, 0))
    #    cv.rectangle(rs, (v.top, v.left), (v.bottom, v.right), (0, 255, 0))
    print(len(ob))

    cv.imwrite('frame626_objects.png', im)
    cv.imshow('grey', rs)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #cv.imshow('grey', im)#
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    print(len(ob))
    

    return ob

def pre_process(ar, old=None):
    """
    finds the different colors in the image
    returns a dict of colors with the amounts
    and returns a dict with all the coords for the colors
    """
    a = dict()
    b = dict()
    c = 0
    d = []

    for row in range(ar.shape[0]):
        for col in range(ar.shape[1]):
            ac = ar[row][col]
            if ac in a:
                a[ac] += 1
                b[ac].append((row, col))
            else:
                a[ac] = 1
                b[ac] = [(row, col)]

            if old != None:
                if ac != old[row][col]:
                    c += 1
                    d.append(old[row][col])

    return a, b, c, d

process_image('./screenshots/frame626')
