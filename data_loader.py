#!/usr/bin/env python3
import sys
import os
from glob import glob
import numpy
import cv2

def data_loader(csvfile):
    nx = 20
    ny = 80
    filenames, plate_numbers = numpy.genfromtxt(csvfile, skip_header = 1, usecols = (0,2), dtype = str, delimiter = ',').T
    X = []
    y = []
    count = 0
    for f, p in zip(filenames, plate_numbers):
        sub_image_paths = get_subimage(f)
        subimages = []
        for im in sub_image_paths:
            image = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (nx, ny), interpolation = cv2.INTER_LINEAR)
            subimages.append(image)

        yy = plate_number_ohe(p)
        if len(yy) != len(subimages): 
            print(f, len(p), len(yy), len(subimages))
            count += 1
            continue
        X += subimages
        y += yy
    print(len(filenames))
    print(count)
    return numpy.array(X), numpy.array(y)

def get_subimage(f):
    directory = os.path.join('segmented_images', f.split('.')[0]) + '/?.png'
    subimages = sorted(glob(directory))
    return subimages

def plate_number_ohe(p):
    string = numpy.array(['a', 'b', 'c', 'd', 'e', 'f', 'g',
                          'h', 'i', 'j', 'k', 'l', 'm', 'n',
                          'o', 'p', 'q', 'r', 's', 't', 'u', 
                          'v', 'w', 'x', 'y', 'z', '1', '2',
                          '3', '4', '5', '6', '7', '8', '9', '0'])
    y = []
    for l in p:
        s = numpy.array([l for _ in string])
        y.append((s == string).astype(int))
    return y





if __name__ == "__main__":
    X, y = data_loader('./PlateImages_only/data.csv')
    print(X.shape, y.shape)
