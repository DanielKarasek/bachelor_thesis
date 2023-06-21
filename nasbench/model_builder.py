from abc import ABC

import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
from nasbench.model_spec import ModelSpec
from nasbench.model_utils import (
  BNConvLayer, operation2layer, Truncate, Projection,
  Conv3x3BnRelu, Conv1x1BnRelu, Identity, Downsample, BasicResBlock)

from nasbench.constants import (
  Constants, NASBENCH201Constants, NASBENCH101Constants)


class StatDecoratorModel(keras.Model, ABC):
  def __init__(self, nas_model: keras.Model):
    super(StatDecoratorModel, self).__init__()
    self.nas_model = nas_model
    self.relu_layers = []
    self.extract_relu_layers(self.nas_model, self.relu_layers)
    self.flatted_relus = [keras.layers.Flatten() for _ in self.relu_layers]

  def extract_relu_layers(self, model_part, relu_layers):
    if isinstance(model_part, keras.Model):
      for layer in model_part.layers:
        self.extract_relu_layers(layer, relu_layers)
    is_activation = (isinstance(model_part, keras.layers.Activation))
    relu_ptr = keras.activations.relu
    if is_activation and (model_part.activation is relu_ptr):
      relu_layers.append(model_part)

  def __call__(self, inputs):
    _ = self.nas_model(inputs)
    with tf.name_scope("distance_calc"):
      K = tf.zeros((inputs.shape[0], inputs.shape[0]))
      for flatted, inp_layer in zip(self.flatted_relus, self.relu_layers):
        x = flatted(inp_layer.output)
        x = tf.where(x > 0., 1., 0.)
        K1 = x @ tf.transpose(x)
        K2 = (1. - x) @ (1. - tf.transpose(x))
        K += K1 + K2
      return K


# TODO: add everywhere training, mask
class NASModel(keras.Model, ABC):
  def __init__(self, cell_spec):
    super(NASModel, self).__init__()
    self.stem = BNConvLayer(NASBENCH101Constants.stem_filter_size, (3, 3))
    channels = NASBENCH101Constants.stem_filter_size
    self.stacks = []
    downsample = False
    for stack_num in range(NASBENCH101Constants.stack_count):
      if stack_num == NASBENCH101Constants.stack_count - 1:
        downsample = True
      self.stacks.append(Stack(cell_spec, channels, downsample))
    self.global_average = keras.layers.GlobalAveragePooling2D(data_format=Constants.data_format)
    self.logits = keras.layers.Dense(Constants.label_count)

  def __call__(self, inputs, training=False, mask=False):
    with tf.name_scope('Stem'):
      x = self.stem(inputs, training, mask)
    for stack in self.stacks:
      x = stack(x)
    with tf.name_scope('final_dense'):
      x = self.global_average(x)
      x = self.logits(x)
      return x

# TODO: input channels might not be needed
class Stack(keras.Model, ABC):
  def __init__(self, cell_spec, output_channels, downsample=True, module_count=3, name="stack"):
    super(Stack, self).__init__()
    self.modules = [Module(cell_spec, output_channels, output_channels) for _ in range(module_count)]
    self.downsample_layer = None
    if downsample:
      self.downsample_layer = keras.layers.MaxPool2D(pool_size=(2, 2),
                                                     strides=(2, 2),
                                                     padding='same',
                                                     data_format=Constants.data_format)

  def __call__(self, x, training=False, mask=False):
    with tf.name_scope('Stack') as _s:
      for module in self.modules:
        x = module(x, training, mask)
      if self.downsample_layer:
        x = self.downsample_layer(x)
      return x


class Module(keras.Model, ABC):
  def __init__(self, cell_spec: ModelSpec, input_channels, output_channels):
    super(Module, self).__init__()
    self.num_vertices = cell_spec.matrix.shape[0]

    vertex_channels = cell_spec.compute_vertex_channels(input_channels,
                                                        output_channels,
                                                        cell_spec.matrix)

    self.vertex_channels = vertex_channels
    self.output_channels = output_channels
    self.input_channels = input_channels
    # build vertex after vertex
    self.vertices = [Vertex(cell_spec, t, vertex_channels) for t in range(1, self.num_vertices - 1)]
    self.concatenate = keras.layers.Concatenate(axis=Constants.channel_axis)

    self.cell_spec = cell_spec

    self.projection = None
    self.add_projection = None
    if cell_spec.matrix[0, -1]:
      self.projection = Projection(output_channels)
      self.add_projection = keras.layers.Add()

  @staticmethod
  def _add_vertex_to_input_lists(vertex_tensor, vertices_input_lists, dst_vector):
    for idx, is_dst in enumerate(dst_vector, -1):
      if is_dst:
        vertices_input_lists[idx].append(vertex_tensor)

  def __call__(self, input_tensor, training=False, mask=False):
    with tf.name_scope('Module'):
      # TODO: demystify this so it isn't just a magic
      mask_matrix = np.asarray(self.cell_spec.matrix, bool)[0:-2]
      vertices_input_lists = [[] for _ in range(self.num_vertices - 1)]
      x = input_tensor
      for vertex_index, dst_vector in enumerate(mask_matrix):
        self._add_vertex_to_input_lists(x, vertices_input_lists, dst_vector)
        x = self.vertices[vertex_index](vertices_input_lists[vertex_index])

      vertices_input_lists[-1].append(x)
      if self.projection:
        vertices_input_lists[-1] = vertices_input_lists[-1][1:]
      x = self.concatenate(vertices_input_lists[-1])

      if self.projection:
        input_projection = self.projection(input_tensor)
        x = self.add_projection([input_projection, x])

      return x


class Vertex(keras.Model, ABC):
  def __init__(self, cell_spec: ModelSpec, vertex_number: int, vertex_channels: np.ndarray):
    super(Vertex, self).__init__()
    adjacency_matrix = cell_spec.matrix
    self.truncated_input = [Truncate(vertex_channels[vertex_number])
                            for src in range(1, vertex_number) if adjacency_matrix[src, vertex_number]]
    if adjacency_matrix[0, vertex_number]:
      self.truncated_input.insert(0, Projection(vertex_channels[vertex_number]))
    self.add = keras.layers.Add()
    self.op = operation2layer[cell_spec.ops[vertex_number]](vertex_channels[vertex_number])

  def __call__(self, input_tensor_list, training=False, mask=False):
    with tf.name_scope('Vertex'):
      assert len(input_tensor_list) == len(self.truncated_input)
      x = [layer(input_tensor) for layer, input_tensor in zip(self.truncated_input, input_tensor_list)]
      if len(x) > 1:
        x = self.add(x)
      else:
        x = x[0]
      return self.op(x)


class NASBench201Vertex(keras.Model, ABC):
  def __init__(self, cell_spec: ModelSpec, vertex_number: int, vertex_channels: int):
    super(NASBench201Vertex, self).__init__()
    self.add = keras.layers.Add()
    self.op = operation2layer[cell_spec.ops[vertex_number]](vertex_channels)

  def __call__(self, input_tensor_list, training=False, mask=False):
    with tf.name_scope('Vertex'):
      if len(input_tensor_list) > 1:
        x = self.add(input_tensor_list)
      else:
        x = input_tensor_list[0]
      return self.op(x)


class NASBench201Module(keras.Model, ABC):
  def __init__(self, cell_spec: ModelSpec, channels: int):
    super(NASBench201Module, self).__init__()
    self.num_vertices = cell_spec.matrix.shape[0]

    # build vertex after vertex
    self.vertices = [NASBench201Vertex(cell_spec, t, channels) for t in range(1, self.num_vertices-1)]
    self.cell_spec = cell_spec

    self.final_add = keras.layers.Add()

  @staticmethod
  def _add_vertex_to_input_lists(vertex_tensor, vertices_input_lists, dst_vector):
    for idx, is_dst in enumerate(dst_vector, -1):
      if is_dst:
        vertices_input_lists[idx].append(vertex_tensor)

  def __call__(self, input_tensor, training=False, mask=False):
    with tf.name_scope('Module'):
      mask_matrix = np.asarray(self.cell_spec.matrix, bool)[0:-2]
      vertices_input_lists = [[] for _ in range(self.num_vertices - 1)]
      x = input_tensor
      for vertex_index, dst_vector in enumerate(mask_matrix):
        self._add_vertex_to_input_lists(x, vertices_input_lists, dst_vector)
        x = self.vertices[vertex_index](vertices_input_lists[vertex_index])

      vertices_input_lists[-1].append(x)
      x = self.final_add(vertices_input_lists[-1])
      return x


class NASBench201Stack(keras.Model, ABC):
  def __init__(self, cell_spec: ModelSpec, channels: int, num_modules: int):
    super(NASBench201Stack, self).__init__()
    self.num_modules = num_modules
    self.modules = [NASBench201Module(cell_spec, channels) for _ in range(num_modules)]

  def __call__(self, input_tensor, training=False, mask=False):
    with tf.name_scope('Stack'):
      x = input_tensor
      for module in self.modules:
        x = module(x, training, mask)
      return x


class NASBench201Model(keras.Model, ABC):
  def __init__(self, cell_spec):
    super(NASBench201Model, self).__init__()
    self.stem = BNConvLayer(NASBENCH201Constants.stem_filter_size, (3, 3))
    channels = NASBENCH201Constants.stem_filter_size
    self.skeleton = []
    for stack_num in range(NASBENCH201Constants.stack_count):
      self.skeleton.append(Stack(cell_spec,
                                 channels,
                                 False,
                                 module_count=NASBENCH201Constants.modules_in_stack))
      if stack_num < NASBENCH201Constants.stack_count - 1:
<<<<<<< HEAD
        self.residual.append(BasicResBlock(channels, stride=(2, 2)))
=======
        self.skeleton.append(BasicResBlock(channels, stride=(2, 2)))
>>>>>>> 9065cc3cb3fc80a960d72274d2dc4fc463996d31

    self.global_average = keras.layers.GlobalAveragePooling2D(data_format=Constants.data_format)
    self.logits = keras.layers.Dense(Constants.label_count)

<<<<<<< HEAD
  def __call__(self, inputs, training, mask):
=======
  def __call__(self, inputs, training=False, mask=False):
>>>>>>> 9065cc3cb3fc80a960d72274d2dc4fc463996d31
    with tf.name_scope('Stem'):
      x = self.stem(inputs, training, mask)
    for stack_idx, module in enumerate(self.skeleton):
      x = module(x)
    with tf.name_scope('final_dense'):
      x = self.global_average(x)
      x = self.logits(x)
      return x
