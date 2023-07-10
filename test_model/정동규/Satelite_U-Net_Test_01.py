import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

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

    outputs = Conv2D(2, 1, activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# 입력 이미지의 크기와 클래스 수 지정
input_shape = (1024, 1024, 3)

# U-Net 모델 생성
model = unet(input_shape)

# 학습된 모델 로드
model.load_weights("unet_model.h5")

# 외부 위성 이미지 파일 경로 지정
image_path = "/path/to/satellite/image.jpg"

# 이미지 로드 및 전처리
image = load_img(image_path, target_size=input_shape[:2])
image = img_to_array(image) / 255.0
image = np.expand_dims(image, axis=0)

# 건물 위치 판별
predictions = model.predict(image)
mask = np.argmax(predictions, axis=-1)
mask = np.squeeze(mask, axis=0)

# 건물 위치 시각화
building_mask = np.where(mask == 1, 255, 0).astype(np.uint8)
building_mask = np.expand_dims(building_mask, axis=-1)
building_mask = np.repeat(building_mask, 3, axis=-1)

# 건물 위치 시각화된 이미지 저장
result_image = np.concatenate([image[0], building_mask], axis=1)
result_image = (result_image * 255).astype(np.uint8)
result_image_path = "/path/to/result/image.jpg"
tf.keras.preprocessing.image.save_img(result_image_path, result_image)
