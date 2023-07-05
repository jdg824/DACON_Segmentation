import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import os

# U-Net 모델 정의
def unet(input_shape):
    # 인코더 부분
    inputs = Input(input_shape)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # 디코더 부분
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(drop5)
    up6 = concatenate([up6, drop4])
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3])
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2])
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# 입력 이미지의 크기 지정
input_shape = (1024, 1024, 3)

# U-Net 모델 생성
model = unet(input_shape)

# 데이터셋 경로 지정
total_images_dir = "C:\\Users\\JW\\Downloads\\open\\train_img"
total_masks_dir = "C:\\Users\\JW\\Downloads\\open\\train_mask"

# 데이터 개수
total_data_count = 7140

# 훈련 데이터 개수
train_data_count = int(total_data_count * 0.6)

# 데이터 인덱스 리스트
data_indices = list(range(total_data_count))

# 데이터 인덱스 섞기
random.seed(42)
random.shuffle(data_indices)

# 훈련 데이터 인덱스
train_indices = data_indices[:train_data_count]

# 검증 데이터 인덱스
val_indices = data_indices[train_data_count:]

datagen = ImageDataGenerator(rescale=1./255)

# 훈련 데이터셋 생성
train_images_dir = [os.path.join(total_images_dir, f"train_image_{idx}.png") for idx in train_indices]
train_masks_dir = [os.path.join(total_masks_dir, f"mask_image_{idx}.png") for idx in train_indices]

train_dataset = datagen.flow_from_directory(
    total_images_dir,  # 단일 디렉토리 경로로 수정
    target_size=input_shape[:2],
    class_mode=None,
    seed=42
)

train_masks_dataset = datagen.flow_from_directory(
    total_masks_dir,  # 단일 디렉토리 경로로 수정
    target_size=input_shape[:2],
    class_mode=None,
    seed=42
)

train_generator = zip(train_dataset, train_masks_dataset)

# 검증 데이터셋 생성
val_images_dir = [os.path.join(total_images_dir, f"train_image_{idx}.png") for idx in val_indices]
val_masks_dir = [os.path.join(total_masks_dir, f"mask_image_{idx}.png") for idx in val_indices]

val_dataset = datagen.flow_from_directory(
    total_images_dir,  # 단일 디렉토리 경로로 수정
    target_size=input_shape[:2],
    class_mode=None,
    seed=42
)

val_masks_dataset = datagen.flow_from_directory(
    total_masks_dir,  # 단일 디렉토리 경로로 수정
    target_size=input_shape[:2],
    class_mode=None,
    seed=42
)

val_generator = zip(val_dataset, val_masks_dataset)

# 모델 학습 설정
model.compile(optimizer='adam', loss='binary_crossentropy')

# 모델 학습
model.fit(train_generator, epochs=10, steps_per_epoch=train_data_count, validation_data=val_generator)

# 학습된 모델 저장
model.save_weights("C:\\Users\\JW\\Downloads\\open\\tran_model")