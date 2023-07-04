# import csv
# import numpy as np
# from PIL import Image
# import os

# # CSV 파일 경로
# csv_file_path = "C:\\Users\\JW\\Downloads\\open\\train.csv"

# # 마스크 이미지 크기
# image_width = 256
# image_height = 256

# # 결과를 저장할 폴더 경로
# output_folder = "C:\\Users\\JW\\Downloads\\open\\train_mask"

# # 폴더가 존재하지 않으면 생성
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # CSV 파일 읽기
# with open(csv_file_path, 'r') as csvfile:
#     reader = csv.reader(csvfile)
#     next(reader)  # 헤더 스킵
#     for idx, row in enumerate(reader):
#         # 픽셀 위치와 레이블 정보 가져오기
#         pixel_info = list(map(int, row[:-1]))  # 마지막 열은 레이블이므로 제외
#         label = int(row[-1])  # 마지막 열이 레이블

#         #빈 마스크 이미지 생성
#         mask_image = np.zeros((image_height, image_width), dtype=np.uint8)

#         # 픽셀 단위로 레이블 할당
#         for i in range(0, len(pixel_info), 2):
#             x, y = pixel_info[i], pixel_info[i + 1]
#             mask_image[y, x] = label

#         # 마스크 이미지 저장
#         mask_image = Image.fromarray(mask_image)
#         mask_image_path = os.path.join(output_folder, f'mask_image_{idx}.png')
#         mask_image.save(mask_image_path)

import csv
import numpy as np
from PIL import Image
import os

# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# CSV 파일 경로
csv_file_path = "C:\\Users\\jdg82\\OneDrive\\바탕 화면\\open\\train.csv"

# 마스크 이미지 크기
image_width = 1024
image_height = 1024

# 결과를 저장할 폴더 경로
output_folder = "C:\\Users\\jdg82\\OneDrive\\바탕 화면\\open\\mask"

# 폴더가 존재하지 않으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# CSV 파일 읽기
with open(csv_file_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # 헤더 스킵
    for idx, row in enumerate(reader):
        # RLE 인코딩 정보 가져오기
        mask_rle = row[-1]  # 마지막 열이 RLE 인코딩 정보

        # RLE 디코딩하여 마스크 이미지 생성
        mask = rle_decode(mask_rle, (image_height, image_width))

        # 마스크 이미지 저장
        mask_image = Image.fromarray(mask * 255)  # 이진 마스크를 0과 255로 변환
        mask_image_path = os.path.join(output_folder, f'mask_image_{idx}.png')
        mask_image.save(mask_image_path)
