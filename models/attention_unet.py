
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

def attention_gate(x, g, filters):
    theta = Conv2D(filters, (1, 1))(x)
    phi = Conv2D(filters, (1, 1))(g)
    add = Add()([theta, phi])
    act = Activation("relu")(add)
    psi = Conv2D(1, (1, 1), activation="sigmoid")(act)
    return Multiply()([x, psi])

def attention_unet(input_shape=(256, 256, 3), num_classes=27):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    p2 = MaxPooling2D()(c2)

    # Bottleneck
    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)

    # Decoder
    u1 = UpSampling2D()(c3)
    att1 = attention_gate(c2, u1, 64)
    concat1 = Concatenate()([u1, att1])
    c4 = Conv2D(128, 3, activation='relu', padding='same')(concat1)

    u2 = UpSampling2D()(c4)
    att2 = attention_gate(c1, u2, 32)
    concat2 = Concatenate()([u2, att2])
    c5 = Conv2D(64, 3, activation='relu', padding='same')(concat2)

    outputs = Conv2D(num_classes, 1, activation='softmax')(c5)

    return Model(inputs, outputs)
