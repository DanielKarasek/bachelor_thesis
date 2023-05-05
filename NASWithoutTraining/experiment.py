from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import List

import bisect

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import scipy.stats as stats
import tensorflow.keras as keras  # type: ignore
import seaborn as sns

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

  def to_list(self):
    return [self.unique_hash, self.log_determinant, self.best_final_accuracy]

@dataclass
class NASStatisticExperimentData:
  __slots__ = ['experiment_list', 'name']
  experiment_list: List[StatisticNASSingleEntryResult]
  name: str

  def to_pandas(self):
    return pd.DataFrame([x.to_list() for x in self.experiment_list], columns=["unique_hash",
                                                                              "log_determinant",
                                                                              "best_final_accuracy"])


  def print_variance(self):
    data = self.to_pandas()
    data["best_final_accuracy"] = data["best_final_accuracy"].apply(lambda x: round(x, 2))
    data_different_init = data.groupby("unique_hash").head(50)
    data_different_init.reset_index(drop=True, inplace=True)
    data_different_init.sort_values(by="best_final_accuracy", ascending=True, inplace=True)
    data_different_init = data_different_init.groupby("best_final_accuracy").agg({"log_determinant": np.std})
    with open("data_different_init.txt", "w") as f:
      print(data_different_init, file=f)
    data_different_data = data.groupby("unique_hash").tail(50)
    data_different_data.reset_index(drop=True, inplace=True)
    data_different_data.sort_values(by="best_final_accuracy", ascending=True, inplace=True)
    data_different_data = data_different_data.groupby("best_final_accuracy").agg({"log_determinant": np.std})
    with open("data_different_data.txt", "w") as f:
      print(data_different_data, file=f)
  def boxplot(self):
    data = self.to_pandas()
    data["best_final_accuracy"] = data["best_final_accuracy"].apply(lambda x: round(x, 2))
    data = data.groupby("unique_hash").tail(50)
    data.reset_index(drop=True, inplace=True)
    data.sort_values(by="best_final_accuracy", ascending=True, inplace=True)
    plt.figure(figsize=(10, 10))
    plt.title("Effect of different data", fontsize=35)
    ax = plt.subplot(1, 1, 1)
    sns.boxplot(y="log_determinant", x="best_final_accuracy", data=data, ax=ax)
    ax.set_ylabel("LHMDS", fontsize=23)
    ax.set_xlabel("Models", fontsize=23)

    ax.set_ylim(1100, 1450)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.savefig("plots/boxplot_different_data.png")
    plt.show()

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

  def save_statistics(self):
    r_squared = self.calculate_r_squared()
    kendal_tau = self.calculate_kendall_tau_statistics()
    with open(f"plots/{self.name}_statistics.txt", "w") as f:
      print(f"R squared: {r_squared}", file=f)
      print(f"Kendal tau: {kendal_tau}", file=f)

  def add_experiment(self, experiment: StatisticNASSingleEntryResult):
    self.experiment_list.append(experiment)

  def save(self, path):
    with open(path, "wb") as output_file:
      pickle.dump(self, output_file)

  def positions_changed_by_log_det(self):
    # sort experiments by log determinant
    self.experiment_list.sort(key=lambda x: x.log_determinant, reverse=False)
    log_determinants = [x.log_determinant for x in self.experiment_list]

    indexes = np.random.choice(range(0, len(self.experiment_list)), 400, replace=False)
    log_determinants_upper = [log_determinants[i]+28 for i in indexes]
    log_determinants_lower = [log_determinants[i]-28 for i in indexes]
    indexes_upper = np.array([bisect.bisect_left(log_determinants, x) for x in log_determinants_upper])
    indexes_lower = np.array([bisect.bisect_right(log_determinants, x) for x in log_determinants_lower])
    index_difference = indexes_upper - indexes_lower
    log_determinants_upper = [log_determinants[i] + 24 for i in indexes]
    log_determinants_lower = [log_determinants[i] - 24 for i in indexes]
    indexes_upper = np.array([bisect.bisect_left(log_determinants, x) for x in log_determinants_upper])
    indexes_lower = np.array([bisect.bisect_right(log_determinants, x) for x in log_determinants_lower])
    index_difference2 = indexes_upper - indexes_lower
    with open(f"plots/{self.name}_statistics.txt", "a") as f:
      print(f"Positions changed by log determinant higher estimate: {np.mean(index_difference)}", file=f)
      print(f"Positions changed by log determinant lower estimate: {np.mean(index_difference2)}", file=f)
      print(f"Positions changed by log determinant higher estimate: {np.mean(index_difference)/len(self.experiment_list)}", file=f)
      print(f"Positions changed by log determinant lower estimate: {np.mean(index_difference2)/len(self.experiment_list)}", file=f)

  @staticmethod
  def load(path, name="") -> NASStatisticExperimentData:
    with open(path, "rb") as input_file:
      me = pickle.load(input_file)
      me.name = name
      return me


  def plot_infered_accuracies(self):
    infered_accuracies = self.get_accuracies_from_log_det()
    true_accuracies = [x.best_final_accuracy for x in self.experiment_list]
    plt.scatter(true_accuracies, infered_accuracies)
    plt.title("LHMDS", fontsize=20)
    plt.plot([0, 1], [0, 1], color="grey")
    plt.xlabel("True accuracy", fontsize = 16)
    plt.ylabel("Predicted Accuracy", fontsize = 16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0.5, 1)
    plt.ylim(0.5, 1)
    from matplotlib import patches

    r_squared = self.calculate_r_squared()
    kendal_tau = self.calculate_kendall_tau_statistics()
    r_squared_text = f"R squared: {r_squared:.3f}"
    kendal_tau_text = f"Kendall tau: {kendal_tau.statistic:.3f}"
    circle = patches.Circle((0.5, 0.5), 0.01, linewidth=1, edgecolor='white', facecolor='white')
    plt.legend([circle, circle],[r_squared_text, kendal_tau_text], loc="lower right", shadow=True, fontsize=16)




    plt.savefig(f"plots/{self.name}_infered_accuracies.png")
    plt.show()


class StatisticsNasExperiment:
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

  def plot_results(self):
    self.experiments.plot_infered_accuracies()

  def save_experiments(self, path: str) -> None:
    self.experiments.save(path)


class VarianceExperiment:
  def __init__(self, search_space):
    self.search_space = search_space
    self.experiments = NASStatisticExperimentData([], "variance_experiment")
    self.train_images = setup_dataset()[:, :, :, np.newaxis]

  def save(self, path):
    self.experiments.save(path)

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

  def run_experiment(self):
    for experiment_idx, model_iterator in enumerate(self.search_space, 1):
      self.multiple_setups_experiment(experiment_idx, model_iterator)
      self.different_train_images_experiment(experiment_idx, model_iterator)
  def get_results_from_model(self, model, model_iterator, train_images=None):
    if train_images is None:
      train_images = self.train_images[:128]
    H_k = model.predict(train_images, batch_size=128)
    _, log_det = np.linalg.slogdet(H_k)
    final_accuracy = self.search_space.get_final_accuracy(model_iterator)
    return log_det, final_accuracy

  def multiple_setups_experiment(self, experiment_idx, model_iterator):
    for i in range(50):
      model = self.setup_model(model_iterator)
      log_det, final_accuracy = self.get_results_from_model(model, model_iterator)
      experiment = StatisticNASSingleEntryResult(model_iterator, log_det, final_accuracy)
      self.experiments.add_experiment(experiment)
      print(f"Experiment number: {experiment_idx} with results:\n\t {experiment}")

  def different_train_images_experiment(self, experiment_idx, model_iterator):
    model = self.setup_model(model_iterator)
    for i in range(50):
      log_det, final_accuracy = self.get_results_from_model(model, model_iterator, self.train_images[i*128:(i+1)*128])
      experiment = StatisticNASSingleEntryResult(model_iterator, log_det, final_accuracy)
      self.experiments.add_experiment(experiment)
      print(f"Experiment number: {experiment_idx} with results:\n\t {experiment}")