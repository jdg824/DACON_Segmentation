from PIL import Image

def crop_image(image, patch_size):
    width, height = image.size
    patches = []
    
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = image.crop((x, y, x+patch_size, y+patch_size))
            patches.append(patch)
    
    return patches

# 1024x1024 이미지 로드
image = Image.open("C:\\Users\\곽연규\\OneDrive\\바탕 화면\\공동 AI 경진대회\\open\\train_img\\TRAIN_0000.png")

# 이미지를 224x224 크기의 패치로 잘라서 리스트에 저장
patch_size = 224
patches = crop_image(image, patch_size)


import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose

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
input_shape = (224, 224, 3)

# U-Net 모델 생성
model = unet(input_shape)

# 모델 컴파일 설정
model.compile(optimizer='adam', loss='binary_crossentropy')

# 데이터셋 경로 설정
train_images_dir = "path/to/train_images_directory"
train_masks_dir = "path/to/train_masks_directory"

# ImageDataGenerator를 사용하여 train 이미지와 mask 이미지를 불러옴
image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# train 이미지와 mask 이미지를 불러와서 배치 단위로 제공
image_generator = image_datagen.flow_from_directory(
    train_images_dir,
    target_size=input_shape[:2],
    class_mode=None,
    batch_size=32,
    seed=42
)

mask_generator = mask_datagen.flow_from_directory(
    train_masks_dir,
    target_size=input_shape[:2],
    class_mode=None,
    batch_size=32,
    seed=42
)

# image_generator와 mask_generator를 결합하여 학습 데이터셋 생성
train_dataset = zip(image_generator, mask_generator)

# 모델 학습
model.fit(train_dataset, epochs=10)

# 학습된 모델 저장
model.save("building_segmentation_model.h5")
