# -*- coding: utf-8 -*-

import csv
import numpy as np
from PIL import Image
import os

# RLE ?��코딩 ?��?��
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# CSV ?��?�� 경로
csv_file_path = "C:\\Users\\JW\\Downloads\\open\\train.csv"

# 마스?�� ?��미�?? ?���?
image_width = 1024
image_height = 1024

# 결과�? ????��?�� ?��?�� 경로
output_folder = "C:\\Users\\JW\\Downloads\\open\\train_mask"

# ?��?���? 존재?���? ?��?���? ?��?��
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# CSV ?��?�� ?���?
with open(csv_file_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # ?��?�� ?��?��
    for idx, row in enumerate(reader):
        # RLE ?��코딩 ?���? �??��?���?
        mask_rle = row[-1]  # 마�??�? ?��?�� RLE ?��코딩 ?���?

        # RLE ?��코딩?��?�� 마스?�� ?��미�?? ?��?��
        mask = rle_decode(mask_rle, (image_height, image_width))

        # 마스?�� ?��미�?? ????��
        mask_image = Image.fromarray(mask * 255)  # ?���? 마스?���? 0�? 255�? �??��
        mask_image_path = os.path.join(output_folder, f'mask_image_{idx}.png')
        mask_image.save(mask_image_path)
