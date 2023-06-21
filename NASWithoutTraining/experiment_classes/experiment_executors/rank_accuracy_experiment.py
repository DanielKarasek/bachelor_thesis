from __future__ import annotations

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import tensorflow.keras as keras  # type: ignore

from NASWithoutTraining.stat_decorator_model import StatDecoratorModel
# todo: get rid of this abbomination
from ..data_setup import setup_dataset
from ..experiment_dataclass import NASStatisticExperimentData, StatisticNASSingleEntryResult


class StatisticsNasExperiment:
  """Executor of the experiments that try to use lhmds to infer the rank and accuracy of given neural network"""
  def __init__(self, search_space, name=""):
    self.search_space = search_space
    self.experiments = NASStatisticExperimentData([], name)
    self.train_images = setup_dataset()[:128, :, :, np.newaxis]

  @staticmethod
  def load(path, name=""):
    statistics_experiment = StatisticsNasExperiment(None, name)
    statistics_experiment.experiments = NASStatisticExperimentData.load(path, name)
    return statistics_experiment

  def run_experiment(self):
    for experiment_idx, model_iterator in enumerate(self.search_space, 1):
      nas_model = self.search_space.get_network(model_iterator)

      keras_inputs = keras.layers.Input((28, 28, 1), 128)
      first_module = StatDecoratorModel(nas_model)(keras_inputs)
      model = keras.Model(inputs=keras_inputs, outputs=first_module)
      model.compile(loss=keras.losses.categorical_crossentropy)

      H_k = model.predict(self.train_images, batch_size=128)
      _, log_det = np.linalg.slogdet(H_k)
      final_accuracy = self.search_space.get_final_accuracy(model_iterator)
      experiment = StatisticNASSingleEntryResult(model_iterator, log_det, final_accuracy)
      self.experiments.add_experiment(experiment)
      print(f"Experiment number: {experiment_idx} with results:\n\t {experiment}")

  def save_experiments(self, path: str) -> None:
    self.experiments.save(path)
