import numpy as np
import tensorflow as tf
import glob
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from PIL import Image
image_files = glob.glob("C:\\Users\\IT\\Desktop\\dacon_image\\ex\\*.png")
image_list=[]
# 이미지 배열을 저장할 리스트 생성



for file in image_files:
    image = Image.open(file)  # 이미지 파일 읽기
    image_array = np.array(image)  # NumPy 배열로 변환)
    image_list.append(image_array)  # 리스트에 이미지 배열 추가



image_array = np.stack(image_list)  # 리스트의 이미지 배열을 하나의 NumPy 배열로 변환

print(image_array)  # 변환된 NumPy 배열의 형태 출력
# 데이터 로드 및 전처리
# train_img = np.load('train_img.npy')
# train_mask = np.load('train_mask.npy')
# val_img = np.load('val_img.npy')
# val_mask = np.load('val_mask.npy')



# train_img = train_img / 255.0  # 이미지 정규화 (0~255 범위를 0~1로 조정)
# train_mask = train_mask / 255.0  # 레이블 데이터 정규화
# val_img = val_img / 255.0
# val_mask = val_mask / 255.0




# # U-Net 모델 정의
# def unet(input_shape):
#     inputs = Input(input_shape)

#     # Contracting Path
#     conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
#     conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

#     conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
#     conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

#     conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
#     conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

#     conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
#     conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
#     drop4 = Dropout(0.5)(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

#     # Bottom
#     conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
#     conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
#     drop5 = Dropout(0.5)(conv5)

#     # Expansive Path
#     up6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
#     merge6 = concatenate([drop4, up6], axis=3)
#     conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
#     conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

#     up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
#     merge7 = concatenate([conv3, up7], axis=3)
#     conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
#     conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

#     up8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
#     merge8 = concatenate([conv2, up8], axis=3)
#     conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
#     conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

#     up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
#     merge9 = concatenate([conv1, up9], axis=3)
#     conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
#     conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

#     # Output
#     outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

#     model = Model(inputs=inputs, outputs=outputs)
#     return model

# # 입력 이미지의 형태 정의
# input_shape = (1024, 1024, 3)

# # U-Net 모델 생성
# model = unet(input_shape)

# # 모델 컴파일
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # 데이터 제너레이터 정의
# class DataGenerator(tf.keras.utils.Sequence):
#     def __init__(self, images, masks, batch_size):
#         self.images = images
#         self.masks = masks
#         self.batch_size = batch_size
        
#     def __len__(self):
#         return int(np.ceil(len(self.images) / self.batch_size))
    
#     def __getitem__(self, index):
#         batch_images = self.images[index * self.batch_size:(index + 1) * self.batch_size]
#         batch_masks = self.masks[index * self.batch_size:(index + 1) * self.batch_size]
        
#         return batch_images, batch_masks

# # 모델 학습
# epochs = 100
# batch_size = 16

# train_generator = DataGenerator(train_img, train_mask, batch_size)
# val_generator = DataGenerator(val_img, val_mask, batch_size)

# model.fit(train_generator, epochs=epochs, validation_data=val_generator)