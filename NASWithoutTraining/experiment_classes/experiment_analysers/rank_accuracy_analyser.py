from __future__ import annotations

import matplotlib.pyplot as plt  # type: ignore
import scipy.stats as stats
import tensorflow.keras as keras  # type: ignore
from matplotlib import patches

from NASWithoutTraining.experiment_classes.experiment_dataclass import NASStatisticExperimentData


class LHMDSAccuracyExperimentsAnalyser:
  """
  This class contains functionolaty to analyse the results of the experiments that tried to use LHMDS to predict
  accuracy of the model and rank it amongst other NN models.
  """
  experiment_data: NASStatisticExperimentData

  def __init__(self, experiment_data: NASStatisticExperimentData):
    self.experiment_data = experiment_data

  def calculate_kendall_tau_statistics(self):
    self.experiment_data.sort_by_final_accuracy()
    log_determinants = [x.log_determinant for x in self.experiment_data.experiment_list]
    best_final_accuracies = [x.best_final_accuracy for x in self.experiment_data.experiment_list]
    return stats.kendalltau(log_determinants, best_final_accuracies)

  def calculate_r_squared(self):
    slope, intercept, r_value, p_value, std_err = self.linear_accuracy_approximation()
    return r_value ** 2

  def save_statistics(self):
    r_squared = self.calculate_r_squared()
    kendal_tau = self.calculate_kendall_tau_statistics()
    with open(f"plots/{self.name}_statistics.txt", "w") as f:
      print(f"R squared: {r_squared}", file=f)
      print(f"Kendal tau: {kendal_tau}", file=f)

  def linear_accuracy_approximation(self):
    """This function approximates best accuracy by linear regression on experiment data"""
    self.experiment_data.sort_by_final_accuracy()
    log_determinants = [x.log_determinant for x in self.experiment_data.experiment_list]
    best_final_accuracies = [x.best_final_accuracy for x in self.experiment_data.experiment_list]
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_determinants, best_final_accuracies)
    return slope, intercept, r_value, p_value, std_err

  def get_accuracies_from_log_det(self):
    # Use linear  approximation to infer accuracy from log_det
    slope, intercept, _, _, _ = self.linear_accuracy_approximation()
    log_determinants = [x.log_determinant for x in self.experiment_data.experiment_list]
    return [slope * x + intercept for x in log_determinants]

  def plot_inferred_accuracies_with_stats(self):
    """
    Plots infered accuracies against true accuracies. On top of that plots R^2 and Kendall tau statistics into
    legend rectangle. Plots is saved into plots folder with name given by 'name' variable of this class.
    :return:
    """
    infered_accuracies = self.get_accuracies_from_log_det()
    true_accuracies = [x.best_final_accuracy for x in self.experiment_data.experiment_list]
    plt.scatter(true_accuracies, infered_accuracies)
    plt.title("LHMDS", fontsize=20)
    plt.plot([0, 1], [0, 1], color="grey")
    plt.xlabel("True accuracy", fontsize = 16)
    plt.ylabel("Predicted Accuracy", fontsize = 16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0.5, 1)
    plt.ylim(0.5, 1)

    r_squared = self.calculate_r_squared()
    kendal_tau = self.calculate_kendall_tau_statistics()
    r_squared_text = f"R squared: {r_squared:.3f}"
    kendal_tau_text = f"Kendall tau: {kendal_tau.statistic:.3f}"
    circle = patches.Circle((0.5, 0.5), 0.01, linewidth=1, edgecolor='white', facecolor='white')
    plt.legend([circle, circle], [r_squared_text, kendal_tau_text], loc="lower right", shadow=True, fontsize=16)

    plt.savefig(f"plots/{self.experiment_data.name}_infered_accuracies.png")
    plt.show()

