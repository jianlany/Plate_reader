# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:09:44 2022

@author: samar
"""

import cv2
import imutils
import numpy as np
import pytesseract
import os
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

folder = "Renamed_PlateImages"
new_folder = "Licence_Plates"
for count, filename in enumerate(os.listdir(folder)):
    dst = f"Image_0{str(count)}.jpg"
    src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
    dst =f"{new_folder}/{dst}"
    
    img = cv2.imread(src,cv2.IMREAD_COLOR)
    height, width, channels = img.shape
    #img = cv2.resize(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    #gray = cv2.bilateralFilter(gray, 13, 15, 15) 

    edged = cv2.Canny(gray, 10, 100) 
    #cv2.imwrite(dst, edged)
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None

    for c in contours:
        
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
     
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        detected = 0
        print ("No contour detected")
        continue
    else:
         detected = 1

    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

    mask = np.zeros(gray.shape,np.uint8)
    new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    new_image = cv2.bitwise_and(img,img,mask=mask)

    cv2.imwrite(dst, new_image)