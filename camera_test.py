# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:32:37 2022

@author: mohr
"""

import cv2 as cv
import time

#cam = cv.VideoCapture(0)
# ret, frame = cam.read()
# ret is true if stuff worked

# cam has 1920x1080 pixels

h = cam.get(cv.CAP_PROP_FRAME_HEIGHT)
w = cam.get(cv.CAP_PROP_FRAME_WIDTH)

# width - height
res = [(1280, 720), (1280, 1024), (1920, 1080), (500, 500), (5000, 5000)]

# thing takes only certain combinations apparently
# 1280x720
# 1280x1024
# 1920x1080
# it seems like it takes closest one if other combinations specified

# dont use cam = cv.VideoCapture(0)
# this superficially works, but has random warnings and is slow as hell

cam = cv.VideoCapture(0, cv.CAP_DSHOW)
for i in res:

    ret = cam.set(cv.CAP_PROP_FRAME_WIDTH,i[0])
    ret = cam.set(cv.CAP_PROP_FRAME_HEIGHT,i[1])
    
    h = cam.get(cv.CAP_PROP_FRAME_HEIGHT)
    w = cam.get(cv.CAP_PROP_FRAME_WIDTH)
    
    print(f"Width: {w} Height: {h} Successfull: {ret}")
    #time.sleep(5)
    ret, frame = cam.read()
    #time.sleep(5)
    cv.imwrite(f"{i[0]}x{i[1]}.jpg", frame)

    #time.sleep(5) -> didnt help
    
cam.release()
# can't set after read ?
# you can set after read perfectly fine, if you dont use 
# am = cv.VideoCapture(0)


# it seems like the camera doesnt change setting right away, you have to
# capture one frame before you see effect
cam = cv.VideoCapture(0, cv.CAP_DSHOW)
c = cam.get(cv.CAP_PROP_CONTRAST) # takes values between 0 and 30 apparently (same as in software)
# 30 is more contrast
s = cam.get(cv.CAP_PROP_SATURATION) # takes values between 0 and 127 (same as in software)
h = cam.get(cv.CAP_PROP_HUE) # takes values between -180 and 180 (same as in software)
g = cam.get(cv.CAP_PROP_GAMMA) # takes values between 20 and 250 (same as in software)
s2 = cam.get(cv.CAP_PROP_SHARPNESS) # takes values between 0 and 60 # whats this, not in software
# Software has brightness, from -127 to 128
b = cam.get(cv.CAP_PROP_BRIGHTNESS()) # takes these values
print(f"Saturation: {s} Contrast: {c}")
cam.release()

# Exposure has 4 settings in camera software
# here we have possible settings for cv.CAP_PROP_EXPOSURE: -4, -5, -6, -7


# this doesnt work -> seems like this fails successfully
# ret = cam.set(cv.CAP_PROP_FRAME_HEIGHT,1024)
# ret = cam.set(cv.CAP_PROP_FRAME_WIDTH,1280)

ret, frame = cam.read()
cv.imshow("show", frame)
cv.waitKey()

#cam.release()

# Testing video view
cam = cv.VideoCapture(0, cv.CAP_DSHOW)
while True:
    ret, frame = cam.read()
    cv.imshow("show", frame)
    if cv.waitKey(1) == ord("q"):
        break
cam.release()
