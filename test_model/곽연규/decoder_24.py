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
    img = np.zeros(shape[0]*shape[1]*3, dtype=np.uint8)  # 24비트 이미지를 위해 *3 추가
    for lo, hi in zip(starts, ends):
        img[lo*3:hi*3] = 255  # 흰색으로 설정 (RGB 값 [255, 255, 255])
    return img.reshape(shape[0], shape[1], 3)  # 24비트 이미지로 reshape

# CSV 파일 경로
csv_file_path = "C:\\Users\\곽연규\\OneDrive\\바탕 화면\\공동 AI 경진대회\\open\\train.csv"

# 마스크 이미지 크기
image_width = 1024
image_height = 1024

# 결과를 저장할 폴더 경로
output_folder = "C:\\Users\\곽연규\\OneDrive\\바탕 화면\\공동 AI 경진대회\\open\\train_mask_24bit"

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
        mask_image = Image.fromarray(mask)
        mask_image_path = os.path.join(output_folder, f'mask_image_{idx}.png')
        mask_image.save(mask_image_path)
