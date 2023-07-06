import numpy as np
import tensorflow as tf
import glob
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from PIL import Image

t_img_file = glob.glob("C:\\Users\\IT\\Desktop\\DACON\\open\\tra_img\\train_img\\*.png")
t_mask_file = glob.glob("C:\\Users\\IT\\Desktop\\DACON\\open\\tra_ma\\train_mask\\*.png")
v_img_file = glob.glob("C:\\Users\\IT\\Desktop\\DACON\\open\\val_im\\val_img\\*.png")
v_mask_file = glob.glob("C:\\Users\\IT\\Desktop\\DACON\\open\\val_ma\\val_ask\\*.png")

t_img=[]
t_mask=[]
v_img=[]
v_mask=[]

# 이미지 배열을 저장할 리스트 생성

for file in t_img_file:
    image = Image.open(file)  # 이미지 파일 읽기
    image_array = np.array(image)  # NumPy 배열로 변환)
    t_img.append(image_array)  # 리스트에 이미지 배열 추가
    
print(t_img.shape)   

for file in t_mask_file:
    image = Image.open(file)  # 이미지 파일 읽기
    image_array = np.array(image)  # NumPy 배열로 변환)
    t_mask.append(image_array)  # 리스트에 이미지 배열 추가

print(t_mask.shape) 

for file in v_img_file:
    image = Image.open(file)  # 이미지 파일 읽기
    image_array = np.array(image)  # NumPy 배열로 변환)
    v_img.append(image_array)  # 리스트에 이미지 배열 추가

print(v_img.shape) 

for file in v_mask_file:
    image = Image.open(file)  # 이미지 파일 읽기
    image_array = np.array(image)  # NumPy 배열로 변환)
    v_mask.append(image_array)  # 리스트에 이미지 배열 추가

print(v_mask.shape) 

# U-Net 모델 정의
def unet(input_shape, num_classes):
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

    outputs = Conv2D(num_classes, 1, activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model

input_shape = (1024, 1024, 3)
num_classes = 2

# U-Net 모델 생성
model = unet(input_shape, num_classes)

# 데이터셋 경로 지정
train_images_dir = "C:\\Users\\IT\\Desktop\\DACON\\open\\tra_img"
train_masks_dir = "C:\\Users\\IT\\Desktop\\DACON\\open\\tra_ma"
val_images_dir = "C:\\Users\\IT\\Desktop\\DACON\\open\\val_im"
val_masks_dir = "C:\\Users\\IT\\Desktop\\DACON\\open\\val_ma"

# 데이터 증강 생성기
datagen = ImageDataGenerator(rescale=1./255)

# 훈련 데이터셋 생성
train_images = datagen.flow_from_directory(
    train_images_dir,
    target_size=input_shape[:2],
    batch_size=16,
    class_mode=None,
    seed=42
)

train_masks = datagen.flow_from_directory(
    train_masks_dir,
    target_size=input_shape[:2],
    batch_size=16,
    class_mode=None,
    seed=42
)

train_generator = zip(train_images, train_masks)

# 검증 데이터셋 생성
val_images = datagen.flow_from_directory(
    val_images_dir,
    target_size=input_shape[:2],
    batch_size=16,
    class_mode=None,
    seed=42
)

val_masks = datagen.flow_from_directory(
    val_masks_dir,
    target_size=input_shape[:2],
    batch_size=16,
    class_mode=None,
    seed=42
)

val_generator = zip(val_images, val_masks)

# 모델 학습 설정
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(train_generator, epochs=10, validation_data=val_generator)

# 학습된 모델 저장
model.save("C:\\Users\\IT\\Desktop\\DACON\\open\\model\\u_net_model.h5")
