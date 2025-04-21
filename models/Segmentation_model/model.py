from tensorflow.keras.applications import ResNet152
from tensorflow.keras import layers, models, regularizers

def build_resnet152_unet(input_shape=(512, 512, 3)):
    base = ResNet152(include_top=False, weights='imagenet', input_shape=input_shape)

    c1 = base.get_layer("conv1_relu").output
    c2 = base.get_layer("conv2_block3_out").output
    c3 = base.get_layer("conv3_block4_out").output
    c4 = base.get_layer("conv4_block6_out").output
    c5 = base.get_layer("conv5_block3_out").output

    def up_block(x, skip, filters):
        x = layers.UpSampling2D()(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, 3, padding="same", activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.3)(x)
        return x

    u6 = up_block(c5, c4, 512)
    u7 = up_block(u6, c3, 256)
    u8 = up_block(u7, c2, 128)
    u9 = up_block(u8, c1, 64)

    u10 = layers.UpSampling2D()(u9)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(u10)

    model = models.Model(inputs=base.input, outputs=outputs)
    return model
