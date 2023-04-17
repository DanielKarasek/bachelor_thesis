from __future__ import annotations

import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


class RegressionExperimentResults:
  def __init__(self):
    self.kendall_taus = []
    self.r2_scores = []
    self.methods_names = []

  def add_result(self, kendall_tau: float, r2_score: float, method_name: str):
    self.kendall_taus.append(kendall_tau)
    self.r2_scores.append(r2_score)
    self.methods_names.append(method_name)

  def save(self, path):
    """pickles this object"""
    with open(path, "wb") as output_file:
      pickle.dump(self, output_file)

  @staticmethod
  def load(path) -> RegressionExperimentResults:
    """loads pickled object"""
    with open(path, "rb") as input_file:
      return pickle.load(input_file)

  def __str__(self):
    return "Kendall Tau stats: {}\nR2 scores: {}\nMethods: {}".format(self.kendall_taus, self.r2_scores, self.methods_names)

  def __repr__(self):
    return self.__str__()


class RegressionExperimentator:
  x_train: np.ndarray
  y_train: np.ndarray

  x_test: np.ndarray
  y_test: np.ndarray

  def __init__(self, data_path):
    data = np.load(data_path, allow_pickle=True)
    from sklearn.preprocessing import PolynomialFeatures
    x = data["x_train"]
    x = PolynomialFeatures(3, include_bias=True).fit_transform(x)
    y = data["y_train"]

    # splits data and saves into attributes
    self.split_data(x, y)

    self.methods = [LinearRegression,
                    SVR,
                    RandomForestRegressor]
    self.methods_names = ["Linear Regression",
                          "SVR",
                          "Random Forest Regression"]
    self.results = RegressionExperimentResults()

  def run_experiment(self):
    for method, method_name in zip(self.methods, self.methods_names):
      kendall_tau, r2 = self.run_experiments_on_single_regressor(method(), method_name)
      self.results.add_result(kendall_tau, r2, method_name)
      print(f"Method: {method_name}\nKendall Tau: {kendall_tau}\nR2: {r2}")
    self.results.save("regressor_results")

  def split_data(self, data_x, data_y, test_size: float = 0.2):
    data = train_test_split(data_x,
                            data_y,
                            test_size=test_size)
    self.x_train, self.x_test, self.y_train, self.y_test = data

  def run_experiments_on_single_regressor(self, regressor, name: str):
    regressor.fit(self.x_train, self.y_train)
    y_pred = regressor.predict(self.x_test)
    kendall_tau = self.get_kendall_tau_stats(y_pred, self.y_test)
    r2 = self.get_r2_score(y_pred, self.y_test)
    y_pred = np.hstack([y_pred, regressor.predict(self.x_train)])
    plt.plot([0,1], [0,1], color="grey")
    plt.scatter(np.hstack([self.y_test, self.y_train]), y_pred)
    plt.xlim(0.5, 1)
    plt.ylim(0.5, 1)
    plt.title(name)
    plt.xlabel("true accuracies")
    plt.ylabel("predicted accuracies")
    plt.savefig(f"regressor_{name}.png")
    plt.show()
    return kendall_tau, r2

  def get_r2_score(self, predicted_y, true_y):
    return scipy.stats.pearsonr(predicted_y, true_y)[0] ** 2

  def get_kendall_tau_stats(self, predicted_y, true_y):
    true_sorted_args = true_y
    predicted_sorted_args = predicted_y
    return scipy.stats.kendalltau(true_sorted_args, predicted_sorted_args)
