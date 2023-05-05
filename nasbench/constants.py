import enum


class Operations(enum.Enum):
  INPUT = 'input'
  OUTPUT = 'output'
  CONV1X1 = 'conv1x1-bn-relu'
  CONV3X3 = 'conv3x3-bn-relu'
  MAXPOOL3X3 = 'maxpool3x3'
  AVGPOOl3X3 = 'avgpool3x3'
  SKIP = 'skip'
  DOWNSAMPLE = 'downsample'
  IDENTITY = 'none'


class NASBENCH201Constants:
  stem_filter_size = 16
  stack_count = 3
  modules_in_stack = 5


class NASBENCH101Constants:
  stem_filter_size = 128
  stack_count = 3
  modules_in_stack = 5


class Constants:
  channel_axis = 3
  data_format = 'channels_last'
  label_count = 10

  MIN_FILTERS = 8
  BN_MOMENTUM = 0.997
  BN_EPSILON = 1e-5
