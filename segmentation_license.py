#!/usr/bin/env python3
import sys
import os
import numpy
from glob import glob
from shutil import copyfile, rmtree
import cv2

def segment_image(image, output_path = 'segmented_images/temp/', debug = False):

    if os.path.exists(output_path): 
        for f in glob(output_path + '/*'):
            os.remove(f)
    else: os.makedirs(output_path)
    if isinstance(image, str):
        original_img = cv2.imread(image,1)
        copyfile(image, os.path.join(output_path, 'plate.png'))
    elif isinstance(image, numpy.ndarray):
        original_img = image
        cv2.imwrite(os.path.join(output_path, 'plate.png'), image)

    # determine assumption space of each letter

    # read image and make it to a gray form to reduce noise

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
    img_gray = cv2.resize(img_gray, (w, h),  interpolation = cv2.INTER_LINEAR)




    _, img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)


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
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations = 3)




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




    white, black, white_max, black_max, black_background = get_black_white_pixels_col(img)
    assert(len(white) == len(black) == w)
    if not black_background: 
        img = invert(img)
        white, black = black, white
        white_max, black_max = black_max, white_max

    cv2.imwrite(os.path.join(output_path, 'processed.png'), img)
    if debug:
        cv2.imshow('processed', img)
        cv2.waitKey(0)

    n = 0
    start = 1
    end = 2
    pieces_num = 1
    sub_images = []
    ww = 50
    hh = 80
    while n < w-1:
        n += 1
        # if the middle line is 0
        if img[int(h/2), n] == 255:
            start = n
            left, right, top, bot = find_char_box(start)
            if right - left > 40: 
                continue
            # sys.exit()
            n = right 
            if top - bot > 0.2*h and right - left > 0.005 * w:
                if debug:
                    print("Box of subimage:")
                    print(left, right, bot, top)
                cj = img[bot:top, left:right]
                h_small, w_small = cj.shape
                scale = 80/h_small
                w_small = int(scale*w_small)
                h_small = int(scale*h_small)
                cj = cv2.resize(cj, (w_small, h_small), interpolation = cv2.INTER_LINEAR)
                if w_small < 50:
                    l = (ww - w_small)//2
                    r = ww - w_small - l
                    cj = numpy.hstack((numpy.zeros((hh, l)),
                                       cj,
                                       numpy.zeros((hh, r))))
                else: 
                    print("subimage is wider than 50 pixels.")
                    print(cj.shape)
                sub_images.append(cj)


    for i, cj in enumerate(sub_images):
        cv2.imwrite('{}/{}.png'.format(output_path, i+1), cj)

    if debug:
        for subimg in sorted(glob(os.path.join(output_path, '?.png'))):
            cv2.imshow('img', cv2.imread(subimg, 1))
            cv2.waitKey(0)

    return sub_images


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
        segment_image(image, os.path.join('segmented_images', os.path.basename(image).split('.')[0]), debug = False)
