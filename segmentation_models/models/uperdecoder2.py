#
# https://www.kaggle.com/code/tt195361/hubmap-training-tf-tpu-efficientnet-b8-640-640/edit
#

from ..backbones.backbones_factory import Backbones
from ._utils import freeze_model
import tensorflow as tf
from tensorflow.keras import models


def ASPP(x, IS_TPU=True, mid_c=320, dilations=[1, 2, 3, 4], out_c=640, debug=False):
    def _aspp_module(x, filters, kernel_size, padding, dilation, groups=1):
        x = tf.keras.layers.ZeroPadding2D(padding=padding)(x)
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            groups=1 if IS_TPU else groups,
            kernel_initializer='he_uniform',
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        return x

    x0 = tf.math.reduce_max(x, axis=(1, 2), keepdims=True)
    x0 = tf.keras.layers.Conv2D(filters=mid_c, kernel_size=1, strides=1, kernel_initializer='he_uniform',
                                use_bias=False)(x0)
    x0 = tf.keras.layers.BatchNormalization(gamma_initializer=tf.constant_initializer(value=0.25))(x0)
    x0 = tf.nn.relu(x0)

    xs = (
            [_aspp_module(x, mid_c, 1, padding=0, dilation=1)] +
            [_aspp_module(x, mid_c, 3, padding=d, dilation=d, groups=4) for d in dilations]
    )

    x0 = tf.image.resize(x0, size=xs[0].shape[1:3])
    x = tf.keras.layers.Concatenate()([x0] + xs)
    x = tf.keras.layers.Conv2D(filters=out_c, kernel_size=1, kernel_initializer='he_uniform', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    if debug:
        return x, x0, xs
    else:
        return x


def upsample(x, concat, target_filters, name, conv2dt_kernel_init_max, relu=True, dropout=0, debug=False):
    filters = concat.shape[-1]
    x_up = tf.keras.layers.Conv2DTranspose(
        filters,  # Number of Convolutional Filters
        kernel_size=4,  # Kernel Size
        strides=2,  # Kernel Steps
        padding='SAME',  # linear scaling
        name=f'Conv2DTranspose_{name}',  # Name of Layer
        kernel_initializer='he_uniform',
        use_bias=False,
    )(x)

    concat = tf.keras.layers.BatchNormalization(
        gamma_initializer=tf.constant_initializer(value=0.25),
        name=f'BatchNormalization_{name}'
    )(concat)
    x = tf.keras.layers.Concatenate(name=f'Concatenate_{name}')([x_up, concat])
    x = tf.nn.relu(x)

    x = tf.keras.layers.Conv2D(target_filters, 3, padding='SAME', kernel_initializer='he_uniform', activation='relu',
                               name=f'Conv2D_1_{name}')(x)
    x = tf.keras.layers.Conv2D(target_filters, 3, padding='SAME', kernel_initializer='he_uniform',
                               name=f'Conv2D_2_{name}')(x)

    if relu:
        x = tf.nn.relu(x)

    x = tf.keras.layers.Dropout(dropout, name=f'Dropout_{name}')(x)

    if debug:
        return x, x_up, concat
    else:
        return x


def FPN(xs, output_channels, last_layer, debug=False):
    def _conv(x):
        x = tf.keras.layers.ZeroPadding2D(padding=1)(x)
        x = tf.keras.layers.Conv2D(output_channels * 2, 3, padding='SAME', kernel_initializer='he_normal', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ZeroPadding2D(padding=1)(x)
        x = tf.keras.layers.Conv2D(output_channels, 3, padding='SAME', kernel_initializer='he_normal')(x)
        x = tf.image.resize(x, size=target_size, method=tf.image.ResizeMethod.BILINEAR)
        x = tf.nn.relu(x)
        return x

    target_size = last_layer.shape[1:3]
    xs = tf.keras.layers.Concatenate()([_conv(x) for x in xs])
    x = tf.keras.layers.Concatenate()([xs, last_layer])

    if debug:
        return x, xs
    else:
        return x


def build_uper_decoder2(
        backbone,
        skip_connection_layers,
        classes=21,
        activation='softmax',
):
    input_ = backbone.input
    up2 = backbone.output
    up3, up4, up5, up6 = \
        ([backbone.get_layer(name=i).output if isinstance(i, str)
          else backbone.get_layer(index=i).output for i in skip_connection_layers])

    dec0 = ASPP(up2)

    dec0 = tf.keras.layers.Dropout(0.30)(dec0)

    dropout_decoder = 0
    dec1 = upsample(dec0, up3, up4.shape[-1] * 4, 'upsample1', 0.02, dropout=dropout_decoder)
    dec2 = upsample(dec1, up4, up5.shape[-1] * 2, 'upsample2', 0.02, dropout=dropout_decoder)
    dec3 = upsample(dec2, up5, up6.shape[-1] * 2, 'upsample3', 0.02)
    dec4 = upsample(dec3, up6, 64, 'upsample4', 0.02)

    dec_fpn = FPN([dec0, dec1, dec2, dec3], 32, dec4)

    x = tf.keras.layers.Dropout(0.10)(dec_fpn)
    x = tf.keras.layers.Conv2D(
        filters=classes,
        kernel_size=1,
        padding='SAME',
        kernel_initializer=tf.random_normal_initializer(0.00, 0.05),
        activation=activation,
        name='Conv2D_3_head'
    )(x)
    IMG_HEIGHT = input_.shape[1]
    IMG_WIDTH = input_.shape[2]
    output = tf.image.resize(x, size=[IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.BILINEAR)

    # create keras model instance
    model = models.Model(input_, output)

    return model


def UPerDecoder2(
        backbone_name='vgg16',
        input_shape=(None, None, 3),
        num_classes=21,
        activation='softmax',
        weights=None,
        encoder_weights='imagenet',
        encoder_freeze=False,
        encoder_features='default',
        **kwargs
):
    backbone = Backbones.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights=encoder_weights,
        include_top=False,
        **kwargs
    )

    if encoder_features == 'default':
        encoder_features = Backbones.get_feature_layers(backbone_name, n=4)

    model = build_uper_decoder2(
        backbone,
        skip_connection_layers=encoder_features,
        classes=num_classes,
        activation=activation,
    )

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model
