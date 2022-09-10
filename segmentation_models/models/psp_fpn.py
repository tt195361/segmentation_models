from keras_applications import get_submodules_from_kwargs

from ._utils import freeze_model, filter_keras_submodules
from ..backbones.backbones_factory import Backbones
from .pspnet import setup_submodules as psp_setup_submodules
from .pspnet import SpatialContextBlock, Conv1x1BnReLU
from .fpn import setup_submodules as fpn_setup_submodules
from .fpn import FPNBlock, DoubleConv3x3BnReLU, Conv3x3BnReLU


def build_psp(
        psp_inpput,
        pooling_type='avg',
        conv_filters=512,
        use_batchnorm=True,
        dropout=None,
):
    x = psp_inpput

    # build spatial pyramid
    x1 = SpatialContextBlock(1, conv_filters, pooling_type, use_batchnorm)(x)
    x2 = SpatialContextBlock(2, conv_filters, pooling_type, use_batchnorm)(x)
    x3 = SpatialContextBlock(3, conv_filters, pooling_type, use_batchnorm)(x)
    x6 = SpatialContextBlock(6, conv_filters, pooling_type, use_batchnorm)(x)

    # aggregate spatial pyramid
    concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x = layers.Concatenate(axis=concat_axis, name='psp_concat')([x, x1, x2, x3, x6])
    x = Conv1x1BnReLU(conv_filters, use_batchnorm, name='aggregation')(x)

    # model regularization
    if dropout is not None:
        x = layers.SpatialDropout2D(dropout, name='spatial_dropout')(x)

    return x


def build_fpn(
        psp_output,
        skip_connection_layers,
        pyramid_filters=256,
        segmentation_filters=128,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True,
        aggregation='sum',
        dropout=None,
):
    x = psp_output
    skips = skip_connection_layers

    # build FPN pyramid
    # p5 = FPNBlock(pyramid_filters, stage=5)(x, skips[0])
    # p4 = FPNBlock(pyramid_filters, stage=4)(p5, skips[1])
    # p3 = FPNBlock(pyramid_filters, stage=3)(p4, skips[2])
    # p2 = FPNBlock(pyramid_filters, stage=2)(p3, skips[3])
    p5 = FPNBlock(pyramid_filters, stage=5)(x, skips[2])
    p4 = FPNBlock(pyramid_filters, stage=4)(p5, skips[1])
    p3 = FPNBlock(pyramid_filters, stage=3)(p4, skips[0])

    # add segmentation head to each
    s5 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage5')(p5)
    s4 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage4')(p4)
    s3 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage3')(p3)
    # s2 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage2')(p2)

    # upsampling to same resolution
    # s5 = layers.UpSampling2D((8, 8), interpolation='nearest', name='upsampling_stage5')(s5)
    # s4 = layers.UpSampling2D((4, 4), interpolation='nearest', name='upsampling_stage4')(s4)
    # s3 = layers.UpSampling2D((2, 2), interpolation='nearest', name='upsampling_stage3')(s3)
    s5 = layers.UpSampling2D((4, 4), interpolation='nearest', name='upsampling_stage5')(s5)
    s4 = layers.UpSampling2D((2, 2), interpolation='nearest', name='upsampling_stage4')(s4)

    # aggregating results
    # if aggregation == 'sum':
    #     x = layers.Add(name='aggregation_sum')([s2, s3, s4, s5])
    # elif aggregation == 'concat':
    #     concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    #     x = layers.Concatenate(axis=concat_axis, name='aggregation_concat')([s2, s3, s4, s5])
    if aggregation == 'sum':
        x = layers.Add(name='aggregation_sum')([s3, s4, s5])
    elif aggregation == 'concat':
        concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1
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

    # model head (define number of output classes)
    x = layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='head_conv',
    )(x)
    x = layers.Activation(activation, name=activation)(x)

    return x


def build_psp_fpn(
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

    psp_input = output_[-1]
    psp_output = build_psp(
        psp_input, psp_pooling_type, psp_conv_filters,
        psp_use_batchnorm, pyramid_dropout)

    # # building decoder blocks with skip connections
    # skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
    #           else backbone.get_layer(index=i).output for i in skip_connection_layers])
    skips = output_[:-1]

    fpn_output = build_fpn(
        psp_output, skips, pyramid_filters, segmentation_filters, classes, activation,
        pyramid_use_batchnorm, pyramid_aggregation, pyramid_dropout)

    # create keras model instance
    model = models.Model(input_, fpn_output)

    return model


def PSP_FPN(
        backbone_name='vgg16',
        input_shape=(None, None, 3),
        classes=21,
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
        include_top=False,
        **kwargs
    )

    # if encoder_features == 'default':
    #     encoder_features = Backbones.get_feature_layers(backbone_name, n=4)

    model = build_psp_fpn(
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
        classes=classes,
        activation=activation,
    )

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model
