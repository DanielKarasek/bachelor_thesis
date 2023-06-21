from abc import ABC

import tensorflow.keras as keras
from nasbench.constants import Constants, Operations
import tensorflow as tf


class BNConvLayer(keras.Model, ABC):
  """Can be used to project """
  def __init__(self, conv_filters, conv_size, stride=(1, 1)):
    super(BNConvLayer, self).__init__()
    self.conv2d = keras.layers.Conv2D(filters=conv_filters,
                                      kernel_size=conv_size,
                                      strides=stride,
                                      use_bias=False,
                                      kernel_initializer=keras.initializers.variance_scaling,
                                      padding='same',
                                      data_format=Constants.data_format,
                                      activation=None)
    self.bn = keras.layers.BatchNormalization(axis=Constants.channel_axis,
                                              momentum=Constants.BN_MOMENTUM,
                                              epsilon=Constants.BN_EPSILON)
    self.activation = keras.layers.Activation("relu")

  def __call__(self, inputs: keras.layers.Layer, training=False, mask=False):
    with tf.name_scope('BN_conv_relu')as _s:
      x = self.conv2d(inputs)
      x = self.bn(x, training)
      return self.activation(x)


class Projection(keras.Model, ABC):
  def __init__(self, channels):
    super(Projection, self).__init__()
    self.projection = BNConvLayer(channels, 1)

  def __call__(self, inputs: keras.layers.Layer, training=False, mask=False):
    with tf.name_scope('Projection'):
      return self.projection(inputs, training, mask)


class Conv3x3BnRelu(keras.Model, ABC):
  def __init__(self, channels, stride=(1, 1), padding="same", data_format=Constants.data_format):
    super(Conv3x3BnRelu, self).__init__()
    self.conv3x3bn_relu = BNConvLayer(channels, (3, 3), stride)

  def __call__(self, inputs: keras.layers.Layer, training=False, mask=False):
    with tf.name_scope('Conv3x3BnRelu')as _s:
      return self.conv3x3bn_relu(inputs, training, mask)


class Conv1x1BnRelu(keras.Model, ABC):
  def __init__(self, channels):
    super(Conv1x1BnRelu, self).__init__()
    self.conv1x1bn_relu = BNConvLayer(channels, (1, 1))

  def __call__(self, inputs: keras.layers.Layer, training=False, mask=False):
    with tf.name_scope('Conv1x1BnRelu')as _s:
      return self.conv1x1bn_relu(inputs, training, mask)


class MaxPool3x3(keras.Model, ABC):
  def __init__(self, _channels, stride = (1, 1), padding="same", data_format=Constants.data_format):
    super(MaxPool3x3, self).__init__()
    self.max_pool3x3 = keras.layers.MaxPool2D(pool_size=(3, 3),
                                              strides=stride,
                                              padding=padding,
                                              data_format=data_format)

  def __call__(self, inputs: keras.layers.Layer, training=False, mask=False):
    with tf.name_scope('MAX_pool3x3')as _s:
      return self.max_pool3x3(inputs)


class AvgPool3x3(keras.Model, ABC):
  def __init__(self, _channels, stride=(1, 1), padding="same", data_format=Constants.data_format):
    super(AvgPool3x3, self).__init__()
    self.avg_pool3x3 = keras.layers.AveragePooling2D(pool_size=(3, 3),
                                                     strides=stride,
                                                     padding=padding,
                                                     data_format=data_format)

  def __call__(self, inputs: keras.layers.Layer, training=False, mask=False):
    with tf.name_scope('AVG_pool3x3')as _s:
      return self.avg_pool3x3(inputs)


class Downsample(keras.Model, ABC):
  def __init__(self, channels, stride=(2, 2), padding="same", data_format=Constants.data_format):
    super(Downsample, self).__init__()
    self.conv1x1 = Conv1x1BnRelu(channels)
    self.avg_pool3x3 = AvgPool3x3(channels, stride, padding=padding, data_format=data_format)

  def __call__(self, inputs: keras.layers.Layer, training=False, mask=False):
    with tf.name_scope('Downsample')as _s:
      x = self.conv1x1(inputs, training, mask)
      return self.avg_pool3x3(x, training, mask)


class BasicResBlock(keras.Model, ABC):
  def __init__(self, channels, stride=2):
    super(BasicResBlock, self).__init__()
<<<<<<< HEAD
=======
    if isinstance(stride, tuple):
      stride = stride[0]
>>>>>>> 9065cc3cb3fc80a960d72274d2dc4fc463996d31
    self.conv1 = Conv3x3BnRelu(channels, stride)
    self.conv2 = Conv3x3BnRelu(channels)
    self.residual = Projection(channels)
    if stride == 2:
      self.residual = Downsample(channels, stride=2)

  def __call__(self, inputs: keras.layers.Layer, training=False, mask=False):
    with tf.name_scope('BasicResBlock')as _s:
      x = self.conv1(inputs, training, mask)
      x = self.conv2(x, training, mask)
      residual = self.residual(inputs, training, mask)
      return x + residual


class Identity(keras.Model, ABC):
  def __init__(self, _channels):
    super(Identity, self).__init__()

  def __call__(self, inputs: keras.layers.Layer, training=False, mask=False):
    with tf.name_scope('Identity'):
      return inputs


class Truncate(keras.Model, ABC):
  def __init__(self, expected_channels):
    super(Truncate, self).__init__()
    self.expected_channels = int(expected_channels)

  def __call__(self, inputs: keras.layers.Layer, training=False, mask=False):
    with tf.name_scope('Truncate'):
      inputs = inputs[:, :, :, :self.expected_channels]
      return inputs


operation2layer = {Operations.CONV1X1.value: Conv1x1BnRelu,
                   Operations.CONV3X3.value: Conv3x3BnRelu,
                   Operations.MAXPOOL3X3.value: MaxPool3x3,
                   Operations.AVGPOOl3X3.value: AvgPool3x3,
<<<<<<< HEAD
                   Operations.SKIP: Identity}
=======
                   Operations.SKIP.value: Identity}
>>>>>>> 9065cc3cb3fc80a960d72274d2dc4fc463996d31
