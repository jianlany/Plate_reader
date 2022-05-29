"""
Created on Tue Apr 20 11:54:13 2022

@author: Elijah White
"""


import cv2 as cv
import sys
import torch
import numpy as np
import math
def grpicshow(FN):
    img = cv.imread(cv.samples.findFile(FN))
    if img is None:
        sys.exit("Could not read the image.")
    cv.imshow("Display window", img)
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    if imgGray is None:
        sys.exit("Could not read the gray image.")
    cv.imshow("Gray Image", imgGray)
    k = cv.waitKey(0)
    if k == ord("s"):
        cv.imwrite(FN, img)
        cv.imwrite("FNgray.PNG", imgGray)
    return(imgGray)

def grpic(FN):
    img = cv.imread(cv.samples.findFile(FN))
    if img is None:
        sys.exit("Could not read the image.")
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    if imgGray is None:
        sys.exit("Could not read the gray image.")
    #cv.imwrite(FN, img)
    #cv.imwrite("FNgray.PNG", imgGray)
    return(imgGray)

def pic(FN):
    img = cv.imread(cv.samples.findFile(FN))
    if img is None:
        sys.exit("Could not read the image.")
    return(img)


def carcolor(FN,X1,Y1,X2,Y2,override):
    if X2 == 0 and Y2 == 0 and override == 0:
        override = 1
    plwidth = X2 - X1
    plheight = Y2 - Y1
    mindim = min(plwidth,plheight)
    X2left = X1 - mindim
    X2right = X2 + mindim
    Y2top = Y1 - mindim
    Y2bottom = Y2 + mindim
    img = cv.imread(cv.samples.findFile(FN))
    dimensions = img.shape
    imgblue = 0
    imggreen = 0
    imgred = 0
    if X2left <0 or X2right>dimensions[1] or Y2top <0 or Y2bottom > dimensions[0] or override == 1 or override ==2:
        if ((X1 > 80 and Y1 > 80) or override == 1) and override !=2:
            if override == 1:
                print("override 1, simplified area")
            else:
                print("not enough space in image, taking simplified area")
            ysimple = Y1 - 75
            imgblue = np.average(img[ysimple:Y1,X1:X1+75,0])
            imggreen = np.average(img[ysimple:Y1,X1:X1+75,1])
            imgred = np.average(img[ysimple:Y1,X1:X1+75,2])
            overridebias = [imgred,imggreen,imgblue]
        else:
            if override == 2:
                print("override 2, X1, Y1, poor accuracy")
            else:
                print("not enough space in image,taking X1,Y1, poor accuracy")
            imgblue = img[Y1,X1,0]
            imggreen =img[Y1,X1,1]
            imgred = img[Y1,X1,2]
            overridebias = [imgred,imggreen,imgblue]
    else:
        #print("normal")
        y1normal = Y1 - mindim
        x1normal = X1- mindim
        x2normal = X2 + mindim
        y2normal = Y2 + mindim
        imgbluefull = np.sum(img[y1normal:y2normal,x1normal:x2normal,0])
        imggreenfull = np.sum(img[y1normal:y2normal,x1normal:x2normal,1])
        imgredfull = np.sum(img[y1normal:y2normal,x1normal:x2normal,2])
        imgblueplate = np.sum(img[Y1:Y2,X1:X2,0])
        imggreenplate = np.sum(img[Y1:Y2,X1:X2,1])
        imgredplate = np.sum(img[Y1:Y2,X1:X2,2])
        imgblue = imgbluefull-imgblueplate
        imggreen = imggreenfull - imggreenplate
        imgred = imgredfull - imgredplate
        fulldim = (y2normal-y1normal)*(x2normal-x1normal)
        platedim = (Y2 - Y1)*(X2-X1)
        dimdim = fulldim-platedim
        imgblue = imgblue/dimdim
        imggreen = imggreen/dimdim
        imgred = imgred/dimdim
        #print(imgblue,imggreen,imgred)
    imgcoloravg = [imgred,imggreen,imgblue]
    imgcolor = ["red","green","blue"]
    maxcol = np.argmax(imgcoloravg)
    mincol = np.argmin(imgcoloravg)
    if imgred > 175 and imggreen > 175 and imgblue > 175:
        biascolor = "white"
    elif imgred < 125 and imggreen<125 and imgblue < 125:
        biascolor = "black"
    elif imgred >= 125 and imggreen >= 125 and imgblue >= 125 and imgred <= 175 and imggreen <= 175 and imgblue <= 175:
        biascolor = "silver"
    elif imgred > 175 and imggreen < 150 and imgblue < 150:
        biascolor = "red"
    elif imgred < 150 and imggreen > 175 and imgblue < 150:
        biascolor = "green"
    elif imgred < 150 and imggreen < 150 and imgblue > 175:
        biascolor = "blue"
    elif imgred < 150 and imggreen > 175 and imgblue > 175:
        biascolor = "blue"
    elif imgred > 175 and imggreen > 175 and imgblue < 150:
        biascolor = "yellow"
    elif imgred > 175 and imggreen < 150 and imgblue > 175:
        biascolor = "purple"
    else:
        biascolor = "can't tell"
    if override != 0:
        biascolor = imgcolor[maxcol]
        print(FN,overridebias,biascolor)
    else:    
        print(FN,imgcoloravg,biascolor)
    return(imgcolor[maxcol],imgcoloravg)
            

#grpicshow("1pin.PNG")
#graypic = grpic("1pin.PNG")

whitecar = pic("carwhite.jpg")
whitecarcolor,whitecarmatrix = carcolor("carwhite.jpg",1146,2107,1616,2348,0)
bluecarcolor,bluecarmatrix = carcolor("carblue.jpg",1254,1862,1476,2013,0)
blackcarcolor,blackcarmatrix = carcolor("carblack.jpg",785,1737,1555,2150,0)
truckredcolor,truckredmatrix = carcolor("truckred.jpg",1401,2438,1785,2658,1)


#use this equation
#carcolor("filename",X1,Y1,X2,Y2,override)
#set override to 0 for normal function. set to 1 for simplified square above plate. set to 2 for X1, Y1 color only

















