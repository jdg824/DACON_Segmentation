# -*- coding: utf-8 -*-

import csv
import numpy as np
from PIL import Image
import os

# RLE ?””ì½”ë”© ?•¨?ˆ˜
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# CSV ?ŒŒ?¼ ê²½ë¡œ
csv_file_path = "C:\\Users\\JW\\Downloads\\open\\train.csv"

# ë§ˆìŠ¤?¬ ?´ë¯¸ì?? ?¬ê¸?
image_width = 1024
image_height = 1024

# ê²°ê³¼ë¥? ????¥?•  ?´?” ê²½ë¡œ
output_folder = "C:\\Users\\JW\\Downloads\\open\\train_mask"

# ?´?”ê°? ì¡´ì¬?•˜ì§? ?•Š?œ¼ë©? ?ƒ?„±
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# CSV ?ŒŒ?¼ ?½ê¸?
with open(csv_file_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # ?—¤?” ?Š¤?‚µ
    for idx, row in enumerate(reader):
        # RLE ?¸ì½”ë”© ? •ë³? ê°?? ¸?˜¤ê¸?
        mask_rle = row[-1]  # ë§ˆì??ë§? ?—´?´ RLE ?¸ì½”ë”© ? •ë³?

        # RLE ?””ì½”ë”©?•˜?—¬ ë§ˆìŠ¤?¬ ?´ë¯¸ì?? ?ƒ?„±
        mask = rle_decode(mask_rle, (image_height, image_width))

        # ë§ˆìŠ¤?¬ ?´ë¯¸ì?? ????¥
        mask_image = Image.fromarray(mask * 255)  # ?´ì§? ë§ˆìŠ¤?¬ë¥? 0ê³? 255ë¡? ë³??™˜
        mask_image_path = os.path.join(output_folder, f'mask_image_{idx}.png')
        mask_image.save(mask_image_path)
