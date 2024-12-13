#!/usr/bin/env python3
import sys
import os
from glob import glob
import torch
import numpy
import cv2
from copy import deepcopy
alphabet_string = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
                   'H', 'I', 'J', 'K', 'L', 'M', 'N',
                   'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
                   'V', 'W', 'X', 'Y', 'Z', '1', '2',
                   '3', '4', '5', '6', '7', '8', '9', '0']

def data_loader(csvfile, device = 'cpu'):
    filenames, plate_numbers = numpy.genfromtxt(csvfile, skip_header = 1, usecols = (0,2), dtype = str, delimiter = ',').T
    X = []
    y = []
    count = 0
    valid_seg = []
    for f, p in zip(filenames, plate_numbers):
        valid_segmentation = True
        p = p.strip()
        sub_image_paths = get_subimage(f)
        batch = []
        for im in sub_image_paths:
            image = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
            h, w = image.shape
            if w > 50:
                print(w, im)
                valid_segmentation = False
                break
            image = image.reshape(1, *image.shape)
            batch.append(image)

        yy = plate_number_ohe(p)
        if len(yy) == len(batch) and valid_segmentation: 
            count += 1
            valid_seg.append(f.split('.')[0])
            X += batch
            y += yy
    assert(len(X) == len(y))
    for _ in range(5):
        X += deepcopy(X)
        y += deepcopy(y)
    X = numpy.array(X, dtype = numpy.float32)
    y = numpy.array(y)
    return X, y

def get_subimage(f):
    directory = os.path.join('segmented_images', f.split('.')[0]) + '/?.png'
    subimages = sorted(glob(directory))
    return subimages

def plate_number_ohe(p):
    y = []
    for l in p:
        i = alphabet_string.index(l.capitalize())
        y.append(i)
    return y





if __name__ == "__main__":
    X, y = data_loader('./PlateImages_only/data.csv')
    print(X.shape, y.shape)
