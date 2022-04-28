#!/usr/bin/env python3
import sys
import torch
import torchvision
import cv2
import re
from glob import glob
from char_recognizer_cnn import Net
from segmentation_license import segment_image
from data_loader import alphabet_string

extract_model_num = lambda f : int(re.match('.*model_(\d+)\.pth', f).group(1))
latest_model = sorted(glob('results/model_*.pth'), key = extract_model_num)[-1]
network = Net()
network_state_dict = torch.load(latest_model)
network.load_state_dict(network_state_dict)
network.eval()

def detect_plate_number(img):
    # plate_image = get_plate_image(img)
    sub_images = segment_image(img, 'output/', debug = False)
    plate_numbers = []
    for img in sub_images:
        h, w = img.shape
        output = network(torch.tensor(img.reshape(1, 1, h, w)).float())
        letter = alphabet_string[output.argmax()]
        plate_numbers.append(letter)

    return ''.join(plate_numbers)


if __name__ == '__main__':
    for path in sys.argv[1:]:
        img = cv2.imread(path, 1)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        plate_number = detect_plate_number(img)
        print(plate_number)

