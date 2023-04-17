from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import List

import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import scipy.stats as stats
import tensorflow.keras as keras  # type: ignore

from NASWithoutTraining.stat_decorator_model import StatDecoratorModel


def per_pixel_normalization(images: np.ndarray):
  images = np.asarray(images, np.float32)
  per_pixel_std = np.std(images, axis=0)
  per_pixel_mean = np.mean(images, axis=0)
  images -= per_pixel_mean
  images /= (per_pixel_std + 1e-14)
  return images


def setup_dataset():
  (train_images, train_labels), _ = keras.datasets.fashion_mnist.load_data()
  train_images = per_pixel_normalization(train_images)
  return train_images


@dataclass
class StatisticNASSingleEntryResult:
  __slots__ = ['unique_hash', 'log_determinant', 'best_final_accuracy']
  unique_hash: str
  log_determinant: float
  best_final_accuracy: float


@dataclass
class NASStatisticExperimentData:
  __slots__ = ['experiment_list']
  experiment_list: List[StatisticNASSingleEntryResult]

  def sort_by_final_accuracy(self):
    self.experiment_list.sort(key=lambda x: x.best_final_accuracy, reverse=True)

  def calculate_kendall_tau_statistics(self):
    self.sort_by_final_accuracy()
    log_determinants = [x.log_determinant for x in self.experiment_list]
    best_final_accuracies = [x.best_final_accuracy for x in self.experiment_list]
    return stats.kendalltau(log_determinants, best_final_accuracies)

  def linear_accuracy_approximation(self):
    """This function approximates best accuracy by linear regression on experiment data"""
    self.sort_by_final_accuracy()
    log_determinants = [x.log_determinant for x in self.experiment_list]
    best_final_accuracies = [x.best_final_accuracy for x in self.experiment_list]
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_determinants, best_final_accuracies)
    return slope, intercept, r_value, p_value, std_err

  def get_accuracies_from_log_det(self):
    # Use linear  aproximation to infer accuracy from log_det
    slope, intercept, _, _, _ = self.linear_accuracy_approximation()
    log_determinants = [x.log_determinant for x in self.experiment_list]
    return [slope * x + intercept for x in log_determinants]

  def calculate_r_squared(self):
    slope, intercept, r_value, p_value, std_err = self.linear_accuracy_approximation()
    return r_value ** 2

  def add_experiment(self, experiment: StatisticNASSingleEntryResult):
    self.experiment_list.append(experiment)

  def save(self, path):
    with open(path, "wb") as output_file:
      pickle.dump(self, output_file)

  @staticmethod
  def load(path) -> NASStatisticExperimentData:
    with open(path, "rb") as input_file:
      return pickle.load(input_file)

  def plot_infered_accuracies(self):
    infered_accuracies = self.get_accuracies_from_log_det()
    true_accuracies = [x.best_final_accuracy for x in self.experiment_list]
    plt.scatter(true_accuracies, infered_accuracies)
    plt.title("Log |H_k| vs Final Accuracy")
    plt.plot([0, 1], [0, 1], color="grey")
    plt.xlabel("True accuracy")
    plt.ylabel("Predicted Accuracy")
    plt.xlim(0.5, 1)
    plt.ylim(0.5, 1)
    plt.savefig("infered_accuracies.png")
    plt.show()


class StatisticsNasExperiment:
  def __init__(self, search_space):
    self.search_space = search_space
    self.experiments = NASStatisticExperimentData([])
    self.train_images = setup_dataset()[:128, :, :, np.newaxis]

  @staticmethod
  def load(path):
    statistics_experiment = StatisticsNasExperiment(None)
    statistics_experiment.experiments = NASStatisticExperimentData.load(path)
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

  def plot_results(self):
    self.experiments.plot_infered_accuracies()

  def save_experiments(self, path: str) -> None:
    self.experiments.save(path)
