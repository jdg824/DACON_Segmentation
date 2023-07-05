import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# U-Net 모델 ?��?��
def unet(input_shape):
    # ?��코더 �?�?
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

    # ?��코더 �?�?
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

# ?��?�� ?��미�???�� ?���? �??��
input_shape = (256, 256, 3)

# U-Net 모델 ?��?��
model = unet(input_shape)

# ?��?��?��?�� 경로 �??��
train_images_dir = 'path/to/train/images'
train_masks_dir = 'path/to/train/masks'
val_images_dir = 'path/to/validation/images'
val_masks_dir = 'path/to/validation/masks'

# ?��?��?�� ?��처리 �? 증강 ?��?��
datagen = ImageDataGenerator(rescale=1./255)

# ?��?�� ?��?��?��?�� ?��?��
train_dataset = datagen.flow_from_directory(
    train_images_dir,
    target_size=input_shape[:2],
    class_mode=None,
    seed=42
)

train_masks_dataset = datagen.flow_from_directory(
    train_masks_dir,
    target_size=input_shape[:2],
    class_mode=None,
    seed=42
)

train_generator = zip(train_dataset, train_masks_dataset)

# �?�? ?��?��?��?�� ?��?��
val_dataset = datagen.flow_from_directory(
    val_images_dir,
    target_size=input_shape[:2],
    class_mode=None,
    seed=42
)

val_masks_dataset = datagen.flow_from_directory(
    val_masks_dir,
    target_size=input_shape[:2],
    class_mode=None,
    seed=42
)

val_generator = zip(val_dataset, val_masks_dataset)

# 모델 ?��?�� ?��?��
model.compile(optimizer='adam', loss='binary_crossentropy')

# 모델 ?��?��
model.fit(train_generator, epochs=10, validation_data=val_generator)

# ?��?��?�� 모델 ????��
model.save_weights('path/to/weights.h5')

print("a")

print("�ȳ��ϼ���")