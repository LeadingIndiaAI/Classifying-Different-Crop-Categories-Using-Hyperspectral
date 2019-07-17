from keras.models import *
from keras.layers import *
from keras.optimizers import *

def cust_loss(y_true, y_pred):
    freq = [2284946, 1667, 6670081, 116814, 688086, 22088, 3270321, 29596, 7746,
            36897, 3762, 146746, 34717, 243931, 501753, 4277, 13961, 765618,
            28723, 6, 2, 3675, 67376, 2283, 15, 274969, 1727367,
            947265, 35354, 5119, 256, 310, 181619, 6923, 3498, 230,
            5760039, 115675, 2095111, 54, 836]

    max_freq = 6670081
    sum = 0
    for k in range(41):
        wc = max_freq / freq[k]
        for i in range(128):
            for j in range(128):
                sum = sum - (y_true[i][j][k] * tf.log(y_pred[i][j][k]) + (1 - y_true[i][j][k]) * tf.log(
                    1 - y_pred[i][j][k])) * wc
    return sum


def unet(classes, input_size=(128, 128, 7)):
    inputs = Input(input_size)
    conv0 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv0)
    conv1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    drop3 = Dropout(0.5)(conv3)

    conv4 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    drop6 = Dropout(0.5)(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop6))  # cahnge conv 3 to conv 6
    merge7 = concatenate([drop3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    drop7 = Dropout(0.5)(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    drop8 = Dropout(0.5)(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(classes, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    out = Conv2D(classes, 3, activation='softmax', padding='same', kernel_initializer='he_normal')(conv9)
    model = Model(input=inputs, output=out)
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model