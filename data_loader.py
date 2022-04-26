#!/usr/bin/env python3
import sys
import os
from glob import glob
import numpy
import cv2
alphabet_string = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
                   'H', 'I', 'J', 'K', 'L', 'M', 'N',
                   'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
                   'V', 'W', 'X', 'Y', 'Z', '1', '2',
                   '3', '4', '5', '6', '7', '8', '9', '0']

def data_loader(csvfile):
    filenames, plate_numbers = numpy.genfromtxt(csvfile, skip_header = 1, usecols = (0,2), dtype = str, delimiter = ',').T
    hh = 80
    ww = 100
    X = []
    y = []
    count = 0
    valid_seg = []
    for f, p in zip(filenames, plate_numbers):
        valid_segmentation = True
        p = p.strip()
        sub_image_paths = get_subimage(f)
        subimages = []
        for im in sub_image_paths:
            image = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
            h, w = image.shape
            scale = 80/h
            image = cv2.resize(image, (int(w*scale), int(h*scale)), interpolation = cv2.INTER_LINEAR)
            h, w = image.shape
            if w < 50:
                left = (ww - w)//2
                right = ww - w - left
                # print(w, left, right)
                # print(numpy.zeros((h, left)).shape)
                # print(numpy.zeros((h, right)).shape)
                image = numpy.hstack((numpy.zeros((h, left)),
                                      image,
                                      numpy.zeros((h, right))))
            elif w >= 50:
                print(w, im)
                valid_segmentation = False
                break
            image = image.reshape(1, 1, hh, ww)
            subimages.append(image)

        yy = plate_number_ohe(p)
        if len(yy) == len(subimages) and valid_segmentation: 
            count += 1
            valid_seg.append(f.split('.')[0])
            X += subimages
            y += yy
    assert(len(X) == len(y))
    print(len(filenames))
    print(count)
    return numpy.array(X), numpy.array(y).reshape(-1, 1)

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
