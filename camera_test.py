# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:32:37 2022

@author: mohr
"""

import cv2 as cv

cam = cv.VideoCapture(0)
ret, frame = cam.read()

cv.imshow("show", frame)
cv.waitKey()