from __future__ import annotations

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import tensorflow.keras as keras  # type: ignore

from NASWithoutTraining.stat_decorator_model import StatDecoratorModel
from NASWithoutTraining.experiment_classes.experiment_dataclass import StatisticNASSingleEntryResult, NASStatisticExperimentData
from NASWithoutTraining.experiment_classes.data_setup import setup_dataset


class VarianceExperiment:
  """
  Executor of the experiment that compares variance lhmds based on different
  initializations and different input data.
  """
  def __init__(self, search_space):
    self.search_space = search_space
    self.different_init = NASStatisticExperimentData([], "different_init")
    self.different_inp_data = NASStatisticExperimentData([], "different_inp_data")
    self.train_images = setup_dataset()[:, :, :, np.newaxis]

  def save(self, path: str):
    self.different_init.save(path + "_different_init")
    self.different_inp_data.save(path + "_different_inp_data")

  @staticmethod
  def load(path: str, name: str = "") -> VarianceExperiment:
    variance_experiment = VarianceExperiment(None)
    variance_experiment.experiments = NASStatisticExperimentData.load(path, name)
    return variance_experiment

  def setup_model(self, model_iterator):
    nas_model = self.search_space.get_network(model_iterator)

    keras_inputs = keras.layers.Input((28, 28, 1), 128)
    first_module = StatDecoratorModel(nas_model)(keras_inputs)
    model = keras.Model(inputs=keras_inputs, outputs=first_module)
    model.compile(loss=keras.losses.categorical_crossentropy)
    return model

  def run_experiment(self, repeats=20):
    for experiment_idx, model_iterator in enumerate(self.search_space, 1):
      self.multiple_setups_experiment(experiment_idx, model_iterator, repeats)
      self.different_train_images_experiment(experiment_idx, model_iterator, repeats)

  def get_results_from_model(self, model, model_iterator, train_images=None):
    if train_images is None:
      train_images = self.train_images[:128]
    H_k = model.predict(train_images, batch_size=128)
    _, log_det = np.linalg.slogdet(H_k)
    final_accuracy = self.search_space.get_final_accuracy(model_iterator)
    return log_det, final_accuracy

  def multiple_setups_experiment(self, experiment_idx, model_iterator, repeats=20):
    for i in range(repeats):
      model = self.setup_model(model_iterator)
      log_det, final_accuracy = self.get_results_from_model(model, model_iterator)
      experiment = StatisticNASSingleEntryResult(model_iterator, log_det, final_accuracy, i)
      self.different_init.add_experiment(experiment)
      print(f"Experiment number: {experiment_idx} with results:\n\t {experiment}")

  def different_train_images_experiment(self, experiment_idx, model_iterator, repeats=20):
    model = self.setup_model(model_iterator)
    for i in range(repeats):
      log_det, final_accuracy = self.get_results_from_model(model, model_iterator,
                                                            self.train_images[i * 128:(i + 1) * 128])
      experiment = StatisticNASSingleEntryResult(model_iterator, log_det, final_accuracy, i)
      self.different_inp_data.add_experiment(experiment)
      print(f"Experiment number: {experiment_idx} with results:\n\t {experiment}")