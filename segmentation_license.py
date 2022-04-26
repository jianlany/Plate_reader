#!/usr/bin/env python3
import sys
import os
import numpy
from glob import glob
from shutil import copyfile, rmtree
import cv2 as cv2

def segment_image(image):
    # determine assumption space of each letter
    segmentation_spacing = 0.99



    # read image and make it to a gray form to reduce noise

    original_img = cv2.imread(image,1)
    h = original_img.shape[0]
    w = original_img.shape[1]
    upper_bp = 0.1
    lower_bp = 0.9
    left_bp = 0.02
    right_bp = 0.98


    img_gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

    # enlarge the pic for easier handling
    scale = 160.0/h
    w = int(scale*w)
    h = int(scale*h)
    print(w, h)
    img_gray = cv2.resize(img_gray, (w, h),  interpolation = cv2.INTER_LINEAR)




    _, img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)


    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    # make a for loop to calculate sum of white and black pixel

    white, black, white_max, black_max, black_background = get_black_white_pixels_col(img)
    if not black_background: 
        img = invert(img)
        white, black = black, white
        white_max, black_max = black_max, white_max


    img[:int(upper_bp*h),:] = 0
    img[int(lower_bp*h):,:] = 0
    img[:,:int(left_bp*w)] = 0
    img[:,int(right_bp*w):] = 0

    kernel = numpy.ones((2,2))
    img = cv2.erode(img, kernel, iterations = 3)
    img = cv2.dilate(img, kernel, iterations = 1)
    kernel = numpy.ones((3,3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)




    # function to split letter


    def find_char_box(start):
        stack = [(h//2, start)]
        visited = set()
        left = w
        right = 0
        top = 0
        bot = h
        while stack:
            i, j = stack.pop()
            if (i,j) in visited: continue
            left =  min(left,  j)
            right = max(right, j)
            top =   max(top,   i)
            bot =   min(bot,   i)
            if right - left > 40:
                return left, right, top, bot
            visited.add((i,j))
            for (ii, jj) in [(i+1, j), (i-1, j),
                             (i, j+1), (i, j-1)]:
                if ii <= h - 1 and ii >= 0 and \
                   jj <= w - 1 and jj >= 0:
                       if (ii,jj) in visited: continue
                       if img[ii,jj] == 255:
                           stack.append((ii,jj))
        return left, right, top, bot


    segmented_path = os.path.join('segmented_images', os.path.basename(image).split('.')[0])
    if os.path.exists(segmented_path): 
        rmtree(segmented_path)
    os.makedirs(segmented_path)
    copyfile(image, os.path.join(segmented_path, os.path.basename(image)))


    white, black, white_max, black_max, black_background = get_black_white_pixels_col(img)
    assert(len(white) == len(black) == w)
    print(black_background)
    if not black_background: 
        img = invert(img)
        white, black = black, white
        white_max, black_max = black_max, white_max

    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    def find_end(zz):
        ending = zz + 1
        for i in range(zz +1, w - 1):
            if black[i] > segmentation_spacing * black_max:
                ending = i
                break
        return ending
    n = 0
    start = 1
    end = 2
    pieces_num = 1
    while n < w-1:
        n += 1
        # if the middle line is 0
        if img[int(h/2), n] == 255:
        # if white[n] > (1 - segmentation_spacing) * white_max:
            start = n
            left, right, top, bot = find_char_box(start)
            if right - left > 40: 
                continue
            # sys.exit()
            n = right 
            if top - bot > 0.2*h and right - left > 0.005 * w:
                print(left, right, bot, top)
                cj = img[bot:top, left:right]
                cv2.imwrite('{}/{}.png'.format(segmented_path, pieces_num), cj)
                pieces_num += 1
                # cv2.imshow('cutChar', cj)
                #cv2.waitKey(0)

    # for subimg in sorted(glob(os.path.join(segmented_path, '?.png'))):
    #     cv2.imshow('img', cv2.imread(subimg, 1))
    #     cv2.waitKey(0)


def get_black_white_pixels_col(img):
    # make a empty list to note each white pixel
    white = []
    # make a empty list to note each white pixel
    black = []
    # and only choose the maximum sum of white of pixel for each column
    white_max = 0
    # and only choose the maximum sum of black of pixel for each column
    black_max = 0
    h, w = img.shape
    black_num = 0
    for i in range(w):
        w_num = 0
        b_num = 0
        for j in range(h):
            if img[j][i] == 255:
                w_num += 1
            else:
                b_num += 1
                black_num += 1
        white_max = max(white_max, w_num)
        black_max = max(black_max, b_num)
        white.append(w_num)
        black.append(b_num)
    if black_num > 0.6*w*h: black_background = True
    else: black_background = False
    return white, black, white_max, black_max, black_background


def invert(img):
    img = (255-img)
    return img


if __name__ == "__main__":
    for image in sys.argv[1:]:
        print(image)
        segment_image(image)
    segment_image('PlateImages_only/188511198478c7.png')
