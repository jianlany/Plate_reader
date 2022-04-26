import cv2.cv2 as cv2
#print(cv2.__version__)

# determine assumption space of each letter
segmentation_spacing = 0.9



# read image and make it to a gray form to reduce noise

img = cv2.imread('test5.jpg',1)
print(img.shape)
"""
cv2.imshow('img', img)
cv2.waitKey(0)
"""

#enlarge = cv2.resize(img, (0,0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
#cv2.namedWindow('img', 0)
#cv2.resizeWindow('img', 400, 300)
cv2.imshow('img', img)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#cv2.namedWindow('img_gray', 0)
#cv2.resizeWindow('img_gray', 400, 300)
cv2.imshow('img_gray',img_gray)
cv2.imwrite('img_gray.png',img_gray)


ret, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
#cv2.imshow('img1', binary)
#img_threshold = img_gray
#cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV, img_threshold)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,1))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,7))
# expansion and corrosion of images
img_gray = cv2.dilate(binary,kernel, anchor=(-1,-1), iterations=2)
img_gray = cv2.erode(img_gray, kernel, anchor=(-1,-1), iterations=4)
img_gray = cv2.dilate(img_gray, kernel, anchor=(-1,-1), iterations=2)
"""
img_gray = cv2.erode(img_gray, kernel2, anchor=(-1,-1), iterations=1)
img_gray = cv2.dilate(img_gray, kernel2, anchor=(-1,-1), iterations=2)
"""
img_gray = cv2.medianBlur(img_gray, 15)
#img_gray = cv2.medianBlur(img_gray,15)
img_threshold = img_gray
cv2.imshow('threshold', img_gray)
cv2.imwrite('threshold.png', img_gray)



#cv2.imshow('threshold', img_thre)

# split letter
# make a empty list to note each white pixel
white = []
# make a empty list to note each white pixel
black = []
h = img_threshold.shape[0]
w = img_threshold.shape[1]
print(w, h)
# and only choose the maximum sum of white of pixel for each column
white_max = 0
# and only choose the maximum sum of black of pixel for each column
black_max = 0


# make a for loop to calculate sum of white and black pixel

for i in range(w):
    w_num = 0
    b_num = 0
    for j in range(h):
        if img_threshold[j][i] == 255:
            w_num += 1
        else:
            b_num += 1
    white_max = max(white_max, w_num)
    black_max = max(black_max, b_num)
    white.append(w_num)
    black.append(b_num)




# to decide which black or white is background
# if Ture then black background
# False then white background
arg = black_max > white_max


# function to split letter
def find_end(zz):
    ending = zz + 1
    for i in range(zz + 1, w - 1):
        if(black[i] if arg else white[i]) > (segmentation_spacing * black_max if arg else segmentation_spacing * white_max):
            ending = i
            break
    return ending


n = 1
start = 1
end = 2
while n < w - 1:
    n += 1
    if(white[n] if arg else black[n]) > ((1 - segmentation_spacing) * white_max if arg else (1 - segmentation_spacing) * black_max):
        start = n
        end = find_end(start)
        n = end
        if end - start > 5:
            print(start, end)
            cj = img_threshold[1:h, start:end]
            #cj = cv2.resize(cj, (15,60))
            cv2.imwrite('./test5_{0}.png'.format(n), cj)
            cv2.imshow('cutChar', cj)
            #cv2.waitKey(0)
