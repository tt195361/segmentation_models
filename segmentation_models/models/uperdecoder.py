from keras_applications import get_submodules_from_kwargs

from ._common_blocks import Conv2dBn
from .pspnet import setup_submodules as psp_setup_submodules
from .fpn import Conv3x3BnReLU, DoubleConv3x3BnReLU
from .fpn import setup_submodules as fpn_setup_submodules
from ._utils import freeze_model, filter_keras_submodules
from ..backbones.backbones_factory import Backbones
import tensorflow as tf
from tensorflow.keras import layers as L

backend = None
layers = None
models = None
keras_utils = None


def get_submodules():
    return {
        'backend': backend,
        'models': models,
        'layers': layers,
        'utils': keras_utils,
    }


def Conv1x1BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=1,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper


def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper


def Resizing(height, width, name):
    def wrapper(x):
        x = L.Lambda(
            lambda x: tf.image.resize(
                x,
                [height, width],
                method=tf.image.ResizeMethod.BILINEAR),
                name=name)(x)
        return x
    return wrapper

def SpatialContextBlock(
        level,
        conv_filters=512,
        pooling_type='avg',
        use_batchnorm=True,
):
    if pooling_type not in ('max', 'avg'):
        raise ValueError('Unsupported pooling type - `{}`.'.format(pooling_type) +
                         'Use `avg` or `max`.')

    Pooling2D = layers.MaxPool2D if pooling_type == 'max' else layers.AveragePooling2D

    pooling_name = 'psp_level{}_pooling'.format(level)
    resizing_name = 'psp_level{}_resizing'.format(level)
    conv_block_name = 'psp_level{}'.format(level)

    def wrapper(input_tensor):
        # extract input feature maps size (h, and w dimensions)
        input_shape = backend.int_shape(input_tensor)
        spatial_size = input_shape[1:3] if backend.image_data_format() == 'channels_last' else input_shape[2:]

        # Compute the kernel and stride sizes according to how large the final feature map will be
        # When the kernel factor and strides are equal, then we can compute the final feature map factor
        # by simply dividing the current factor by the kernel or stride factor
        # The final feature map sizes are 1x1, 2x2, 3x3, and 6x6.
        pool_size = up_size = [spatial_size[0] // level, spatial_size[1] // level]

        height = spatial_size[0]
        width = spatial_size[1]

        x = Pooling2D(pool_size, strides=pool_size, padding='same', name=pooling_name)(input_tensor)
        x = Resizing(height, width, name=resizing_name)(x)
        x = Conv3x3BnReLU(conv_filters, use_batchnorm, name=conv_block_name)(x)
        return x

    return wrapper


def build_ppm(
        input,
        pooling_type='avg',
        conv_filters=512,
        ppm_out_filters=256,
        use_batchnorm=True,
):
    x = input

    # build spatial pyramid
    x1 = SpatialContextBlock(1, conv_filters, pooling_type, use_batchnorm)(x)
    x2 = SpatialContextBlock(2, conv_filters, pooling_type, use_batchnorm)(x)
    x3 = SpatialContextBlock(3, conv_filters, pooling_type, use_batchnorm)(x)
    x6 = SpatialContextBlock(6, conv_filters, pooling_type, use_batchnorm)(x)

    # aggregate spatial pyramid
    concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x = layers.Concatenate(axis=concat_axis, name='psp_concat')([x, x1, x2, x3, x6])
    x = Conv3x3BnReLU(ppm_out_filters, use_batchnorm, name='aggregation')(x)

    return x


def FPNBlock(pyramid_filters, stage):
    block_name = 'fpn_stage_p{}'.format(stage)
    # conv0_name = 'fpn_stage_p{}_pre_conv'.format(stage)
    # conv1_name = 'fpn_stage_p{}_conv'.format(stage)
    add_name = 'fpn_stage_p{}_add'.format(stage)
    up_name = 'fpn_stage_p{}_upsampling'.format(stage)

    channels_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor, skip):
        # if input tensor channels not equal to pyramid channels
        # we will not be able to sum input tensor and skip
        # so add extra conv layer to transform it
        # input_filters = backend.int_shape(input_tensor)[channels_axis]
        # if input_filters != pyramid_filters:
        #     input_tensor = layers.Conv2D(
        #         filters=pyramid_filters,
        #         kernel_size=(1, 1),
        #         kernel_initializer='he_uniform',
        #         name=conv0_name,
        #     )(input_tensor)

        skip = Conv1x1BnReLU(
            filters=pyramid_filters,
            use_batchnorm=True,
            name=block_name,
        )(skip)

        x = layers.UpSampling2D(
            (2, 2), interpolation='bilinear', name=up_name
        )(input_tensor)
        x = layers.Add(name=add_name)([x, skip])

        return x

    return wrapper


def build_fpn(
        ppm_output,
        skips,
        pyramid_filters=256,
        segmentation_filters=128,
        use_batchnorm=True,
        aggregation='sum',
        dropout=None,
):
    x = ppm_output

    # build FPN pyramid
    p5 = FPNBlock(pyramid_filters, stage=5)(x, skips[2])
    p4 = FPNBlock(pyramid_filters, stage=4)(p5, skips[1])
    p3 = FPNBlock(pyramid_filters, stage=3)(p4, skips[0])
    # p2 = FPNBlock(pyramid_filters, stage=2)(p3, skips[3])

    # add segmentation head to each
    s5 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage5')(p5)
    s4 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage4')(p4)
    s3 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage3')(p3)
    # s2 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage2')(p2)

    # upsampling to same resolution
    # s5 = layers.UpSampling2D((8, 8), interpolation='bilinear', name='upsampling_stage5')(s5)
    # s4 = layers.UpSampling2D((4, 4), interpolation='bilinear', name='upsampling_stage4')(s4)
    # s3 = layers.UpSampling2D((2, 2), interpolation='bilinear', name='upsampling_stage3')(s3)
    s5 = layers.UpSampling2D((4, 4), interpolation='bilinear', name='upsampling_stage5')(s5)
    s4 = layers.UpSampling2D((2, 2), interpolation='bilinear', name='upsampling_stage4')(s4)

    # aggregating results
    if aggregation == 'sum':
        # x = layers.Add(name='aggregation_sum')([s2, s3, s4, s5])
        x = layers.Add(name='aggregation_sum')([s3, s4, s5])
    elif aggregation == 'concat':
        concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        # x = layers.Concatenate(axis=concat_axis, name='aggregation_concat')([s2, s3, s4, s5])
        x = layers.Concatenate(axis=concat_axis, name='aggregation_concat')([s3, s4, s5])
    else:
        raise ValueError('Aggregation parameter should be in ("sum", "concat"), '
                         'got {}'.format(aggregation))

    if dropout:
        x = layers.SpatialDropout2D(dropout, name='pyramid_dropout')(x)

    # final stage
    x = Conv3x3BnReLU(segmentation_filters, use_batchnorm, name='final_stage')(x)
    # x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='final_upsampling')(x)
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear', name='final_upsampling')(x)

    return x


def build_uper_decoder(
        backbone,
        skip_connection_layers,
        psp_pooling_type='avg',
        psp_conv_filters=512,
        psp_use_batchnorm=True,
        pyramid_filters=256,
        segmentation_filters=128,
        pyramid_use_batchnorm=True,
        pyramid_dropout=None,
        pyramid_aggregation='sum',
        classes=21,
        activation='softmax',
):
    input_ = backbone.input
    output_ = backbone.output

    ppm_input = output_[-1]
    ppm_output = build_ppm(
        ppm_input, psp_pooling_type, psp_conv_filters,
        pyramid_filters, psp_use_batchnorm)

    # building decoder blocks with skip connections
    # skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
    #           else backbone.get_layer(index=i).output for i in skip_connection_layers])
    skips = output_[:-1]

    fpn_output = build_fpn(
        ppm_output, skips, pyramid_filters, segmentation_filters, pyramid_use_batchnorm,
        pyramid_aggregation, pyramid_dropout)
    # model head (define number of output classes)
    x = layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='head_conv',
    )(fpn_output)
    x = layers.Activation(activation, name=activation)(x)

    # create keras model instance
    model = models.Model(input_, x)

    return model


def UPerDecoder(
        backbone_name='vgg16',
        input_shape=(None, None, 3),
        num_classes=21,
        activation='softmax',
        weights=None,
        encoder_weights='imagenet',
        encoder_freeze=False,
        encoder_features='default',
        pyramid_block_filters=256,
        pyramid_use_batchnorm=True,
        pyramid_aggregation='concat',
        pyramid_dropout=None,
        psp_conv_filters=512,
        psp_pooling_type='avg',
        psp_use_batchnorm=True,
        **kwargs
):
    global backend, layers, models, keras_utils
    submodule_args = filter_keras_submodules(kwargs)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(submodule_args)
    psp_setup_submodules(submodule_args)
    fpn_setup_submodules(submodule_args)

    backbone = Backbones.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights=encoder_weights,
        **kwargs
    )

    # if encoder_features == 'default':
    #     encoder_features = Backbones.get_feature_layers(backbone_name, n=4)

    model = build_uper_decoder(
        backbone,
        skip_connection_layers=encoder_features,
        psp_pooling_type=psp_pooling_type,
        psp_conv_filters=psp_conv_filters,
        psp_use_batchnorm=psp_use_batchnorm,
        pyramid_filters=pyramid_block_filters,
        segmentation_filters=pyramid_block_filters // 2,
        pyramid_use_batchnorm=pyramid_use_batchnorm,
        pyramid_dropout=pyramid_dropout,
        pyramid_aggregation=pyramid_aggregation,
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
