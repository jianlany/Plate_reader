# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 17:22:48 2022

@author: samar
"""

import os
import cv2
import numpy as np
import imutils



new_folder = "Cropped_PlateImages"
folder = "Renamed_PlateImages"
for count, filename in enumerate(os.listdir(folder)):
    dst = f"Image_{str(count)}.jpg"
    src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
    dst =f"{new_folder}/{dst}"
    img = cv2.imread(src,cv2.IMREAD_COLOR)
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    gray = cv2.cvtColor(image_sharp, cv2.COLOR_BGR2GRAY) 
    gray = cv2.bilateralFilter(gray, 13, 55, 55) 
    gray_1 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY);
    cv2.imwrite(dst, image_sharp)   
        # rename() function will
        # rename all the files
    