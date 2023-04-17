from abc import ABC

import tensorflow.keras as keras
import tensorflow as tf


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
        K2 = (1.-x) @ (1.-tf.transpose(x))
        K += K1 + K2
      return K


