# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 17:22:48 2022

@author: samar
"""
from __future__ import print_function
import os
import cv2
import numpy as np
import imutils
import argparse
import random as rng


rng.seed(12345)

def thresh_callback(val):
    threshold = val
    # Detect edges using Canny
    canny_output = cv2.Canny(gray, threshold, threshold * 2)
    # Find contours
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    Area = []
    x_max = 100
    y_max = 100
    w_max = 100
    h_max = 100
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            Area.append(cv2.contourArea(c))
    
    for i in range(len(contours)):
        peri = cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], 0.04 * peri, True)
        
        if len(approx) == 4:
			# compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
			# a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
            if ar >= 0.85 and cv2.contourArea(contours[i]) >= Area[0]:
                x_max = x
                y_max = y
                w_max = w
                h_max = h
                color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
                #cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
                #new_image = cv2.bitwise_and(gray,gray,mask=mask)
    return img[y_max:y_max+h_max,x_max:x_max+w_max], x_max, y_max, w_max, h_max


folder = "Renamed_PlateImages"
new_folder = "GrayImages_PlateImages"
for count, filename in enumerate(os.listdir(folder)):
    dst = f"Image_{str(count)}.jpg"
    src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
    dst =f"{new_folder}/{dst}"
    img = cv2.imread(src,cv2.IMREAD_COLOR)
    height, width, channels = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    #gray = cv2.bilateralFilter(gray, 13, 15, 15) 
    gray_1 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    new_image, x, y, w, h = thresh_callback(100)
    #new_image[y:y+h,x:x+w] = 255
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    image_sharp = cv2.filter2D(src=new_image, ddepth=-1, kernel=kernel)
    cv2.imwrite(dst, image_sharp)   
    print(count)
        # rename() function will
        # rename all the files
    