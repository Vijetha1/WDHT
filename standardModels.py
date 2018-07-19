# -*- coding: utf-8 -*-
import numpy as np
from customLayers import crosschannelnormalization
from customLayers import Softmax4D
from customLayers import splittensor
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import SGD
import pdb
import h5py
import pdb

#The below imports for vgg 19
# from __future__ import print_function
# from __future__ import absolute_import

# import warnings

# from ..models import Model
# from ..layers import Flatten
# from ..layers import Dense
# from ..layers import Input
# from ..layers import Conv2D
# from ..layers import MaxPooling2D
# from ..layers import GlobalAveragePooling2D
# from ..layers import GlobalMaxPooling2D
# from ..engine.topology import get_source_inputs
# from ..utils import layer_utils
# from ..utils.data_utils import get_file
# from .. import backend as K
# from .imagenet_utils import decode_predictions
# from .imagenet_utils import preprocess_input
# from .imagenet_utils import _obtain_input_shape

def AlexNet(weights_path=None, retainTop = True):
    inputs = Input(shape=(3, 227, 227))

    conv_1 = Conv2D(96, (11, 11), strides=(4, 4), activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    #pdb.set_trace()
    conv_2 = crosschannelnormalization(name='convpool_1')(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    concat1 = Concatenate(axis=1, name='conv_2')
    conv_2 = concat1([
                       Conv2D(128, (5, 5), activation='relu', name='conv_2_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_2)
                       ) for i in range(2)])

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    #conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, (3, 3), activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    concat2 = Concatenate(axis=1, name='conv_4')
    conv_4 = concat2([
                       Conv2D(192, (3, 3), activation='relu', name='conv_4_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_4)
                       ) for i in range(2)])

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    concat3 = Concatenate(axis=1, name='conv_5')
    conv_5 = concat3([
                       Conv2D(128, (3, 3), activation='relu', name='conv_5_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_5)
                       ) for i in range(2)])

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)
    dense_1 = Flatten(name='flatten')(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(1000, name='dense_3')(dense_3)
    prediction = Activation('softmax', name='softmax')(dense_3)
    
    tempModel = Model(inputs=[inputs], outputs=[prediction])
    
    tempModel.load_weights(weights_path)
    if not retainTop:
      model = Model(inputs=[inputs], outputs=[dense_2])
      lastLayer = dense_2
    else:
      model = tempModel
      lastLayer = prediction
    firstLayer = inputs
    return model, firstLayer, lastLayer

def vgg19(weights_path='./pretrainedWeights/vgg19_weights_th_dim_ordering_th_kernels.h5', retainTop = False):
    # -*- coding: utf-8 -*-
    """VGG19 model for Keras.
    # Reference
    - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
    """



    #WEIGHTS_PATH = 
    #WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


    #def VGG19(include_top=True, weights='imagenet',
    #      input_tensor=None, input_shape=None,
    #      pooling=None,
    #      classes=1000):
    """Instantiates the VGG19 architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.
    {"image_data_format": "channels_first", "epsilon": 1e-07, "floatx": "float32", "backend": "theano"}
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or 'imagenet' (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    # if weights not in {'imagenet', None}:
    #     raise ValueError('The `weights` argument should be either '
    #                      '`None` (random initialization) or `imagenet` '
    #                      '(pre-training on ImageNet).')

    # if weights == 'imagenet' and include_top and classes != 1000:
    #     raise ValueError('If using `weights` as imagenet with `include_top`'
    #                      ' as true, `classes` should be 1000')
    # # Determine proper input shape
    # input_shape = _obtain_input_shape(input_shape,
    #                                   default_size=224,
    #                                   min_size=48,
    #                                   data_format=K.image_data_format(),
    #                                   require_flatten=include_top,
    #                                   weights=weights)

    # if input_tensor is None:
    #     img_input = Input(shape=input_shape)
    # else:
    #     if not K.is_keras_tensor(input_tensor):
    #         img_input = Input(tensor=input_tensor, shape=input_shape)
    #     else:
    #         img_input = input_tensor
    classes = 1000
    img_input = Input(shape=(3, 227, 227))
    # Block 1
    conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    conv_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(conv_1)
    max_1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv_2)

    # Block 2
    conv_3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(max_1)
    conv_4 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(conv_3)
    max_2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv_4)

    # Block 3
    conv_5 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(max_2)
    conv_6 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(conv_5)
    conv_7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(conv_6)
    conv_8 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(conv_7)
    max_3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv_8)

    # Block 4
    conv_9 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(max_3)
    conv_10 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(conv_9)
    conv_11 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(conv_10)
    conv_12 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(conv_11)
    max_4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv_12)

    # Block 5
    conv_13 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(max_4)
    conv_14 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(conv_13)
    conv_15 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(conv_14)
    conv_16 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(conv_15)
    max_5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv_16)
    flat = Flatten(name='flatten')(max_5)
    dense_1 = Dense(4096, activation='relu', name='fc1')(flat)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='fc2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    prediction = Dense(classes, activation='softmax', name='predictions')(dense_3)
    tempModel = Model(img_input, prediction, name='vgg19')

    tempModel.load_weights(weights_path)
    if not retainTop:
      model = Model(inputs=[img_input], outputs=[dense_1])
      lastLayer = dense_1
    else:
      model = tempModel
      lastLayer = prediction
    firstLayer = img_input
    return model, firstLayer, lastLayer

        # Classification block

    # else:
    #     if pooling == 'avg':
    #         x = GlobalAveragePooling2D()(x)
    #     elif pooling == 'max':
    #         x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    # if input_tensor is not None:
    #     inputs = get_source_inputs(input_tensor)
    # else:
    #     inputs = img_input
    # Create model.
    
    # weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #                                 weights_path,
    #                                 cache_subdir='models',
    #                                 file_hash='253f8cb515780f3b799900260a226db6')
    #model.load_weights(weights_path)

    # # load weights
    # if weights == 'imagenet':
    #     if include_top:
    #         weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5',
    #                                 WEIGHTS_PATH,
    #                                 cache_subdir='models',
    #                                 file_hash='cbe5617147190e668d6c5d5026f83318')
    #     else:
    #         weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #                                 WEIGHTS_PATH_NO_TOP,
    #                                 cache_subdir='models',
    #                                 file_hash='253f8cb515780f3b799900260a226db6')
    #     model.load_weights(weights_path)
    #     if K.backend() == 'theano':
    #         layer_utils.convert_all_kernels_in_model(model)

    #     if K.image_data_format() == 'channels_first':
    #         if include_top:
    #             maxpool = model.get_layer(name='block5_pool')
    #             shape = maxpool.output_shape[1:]
    #             dense = model.get_layer(name='fc1')
    #             layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

    #         if K.backend() == 'tensorflow':
    #             warnings.warn('You are using the TensorFlow backend, yet you '
    #                           'are using the Theano '
    #                           'image data format convention '
    #                           '(`image_data_format="channels_first"`). '
    #                           'For best performance, set '
    #                           '`image_data_format="channels_last"` in '
    #                           'your Keras config '
    #                           'at ~/.keras/keras.json.')
    #return model