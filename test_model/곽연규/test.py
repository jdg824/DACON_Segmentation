# -*- coding: utf-8 -*-

import csv
import numpy as np
from PIL import Image
import os

# RLE ?μ½λ© ?¨?
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# CSV ??Ό κ²½λ‘
csv_file_path = "C:\\Users\\JW\\Downloads\\open\\train.csv"

# λ§μ€?¬ ?΄λ―Έμ?? ?¬κΈ?
image_width = 1024
image_height = 1024

# κ²°κ³Όλ₯? ????₯?  ?΄? κ²½λ‘
output_folder = "C:\\Users\\JW\\Downloads\\open\\train_mask"

# ?΄?κ°? μ‘΄μ¬?μ§? ??Όλ©? ??±
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# CSV ??Ό ?½κΈ?
with open(csv_file_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # ?€? ?€?΅
    for idx, row in enumerate(reader):
        # RLE ?Έμ½λ© ? λ³? κ°?? Έ?€κΈ?
        mask_rle = row[-1]  # λ§μ??λ§? ?΄?΄ RLE ?Έμ½λ© ? λ³?

        # RLE ?μ½λ©??¬ λ§μ€?¬ ?΄λ―Έμ?? ??±
        mask = rle_decode(mask_rle, (image_height, image_width))

        # λ§μ€?¬ ?΄λ―Έμ?? ????₯
        mask_image = Image.fromarray(mask * 255)  # ?΄μ§? λ§μ€?¬λ₯? 0κ³? 255λ‘? λ³??
        mask_image_path = os.path.join(output_folder, f'mask_image_{idx}.png')
        mask_image.save(mask_image_path)
