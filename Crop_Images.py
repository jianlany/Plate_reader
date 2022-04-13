# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 17:22:48 2022

@author: samar
"""

import os
import cv2
import numpy as np

def BrightnessContrast(brightness=0):
    
    # getTrackbarPos returns the current
    # position of the specified trackbar.
    brightness = cv2.getTrackbarPos('Brightness',
                                    'GEEK')
     
    contrast = cv2.getTrackbarPos('Contrast',
                                  'GEEK')
 
    effect = controller(img, brightness,
                        contrast)
 
    # The function imshow displays an image
    # in the specified window
    return effect
 
def controller(img, brightness=255,
               contrast=127):
   
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
 
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
 
    if brightness != 0:
 
        if brightness > 0:
 
            shadow = brightness
 
            max = 255
 
        else:
 
            shadow = 0
            max = 255 + brightness
 
        al_pha = (max - shadow) / 255
        ga_mma = shadow
 
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(img, al_pha,
                              img, 0, ga_mma)
 
    else:
        cal = img
 
    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)
 
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha,
                              cal, 0, Gamma)
 
    # putText renders the specified text string in the image.
    cv2.putText(cal, 'B:{},C:{}'.format(brightness,
                                        contrast), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
 
    return cal

new_folder = "Cropped_PlateImages"
folder = "Renamed_PlateImages"
for count, filename in enumerate(os.listdir(folder)):
    dst = f"Image_{str(count)}.jpg"
    src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
    dst =f"{new_folder}/{dst}"    
    img = cv2.imread(src,cv2.IMREAD_COLOR)
    cv2.namedWindow('GEEK')
    cv2.createTrackbar('Brightness',
                       'GEEK', 255, 2 * 255,
                       BrightnessContrast)
     
    # Contrast range -127 to 127
    cv2.createTrackbar('Contrast', 'GEEK',
                       127, 2 * 127,
                       BrightnessContrast) 
 
     
    BrightnessContrast(0)

    