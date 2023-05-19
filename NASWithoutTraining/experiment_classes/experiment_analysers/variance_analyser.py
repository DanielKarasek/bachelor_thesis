"""
This module contains wrapper classes for experiments using LHMDS statistics.
"""

from __future__ import annotations

from bisect import bisect

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import seaborn as sns
import tensorflow.keras as keras  # type: ignore

from ..experiment_dataclass import NASStatisticExperimentData


class LHMDSStochasticityExperimentsAnalyser:
  """
  This class contains functionality which can be used to examine the influence of different stochastic factors
  on the LHMDS statistic.
  """

  experiment_data: NASStatisticExperimentData

  def __init__(self, experiment_data: NASStatisticExperimentData):
    self.experiment_data = experiment_data

  def lhmds_stds_by_model(self,
                          save_file_name: str = "lhmd_stochasticity_variances.txt"):
    """
    Calculates the standard deviation of the LHMDS statistic for each unique model in the dataset.
    :param save_file_name: file name to save the results to. Empty string will lead to no saving.
    :return: results as a pandas dataframe
    """
    pd_data = self.experiment_data.to_pandas()
    pd_data["best_final_accuracy"] = pd_data["best_final_accuracy"].apply(lambda x: round(x, 2))
    pd_data.reset_index(drop=True, inplace=True)
    pd_data.sort_values(by="best_final_accuracy", ascending=True, inplace=True)
    pd_data = pd_data.groupby("best_final_accuracy").agg({"log_determinant": np.std})
    if not save_file_name:
      return pd_data.copy()
    with open(f"data/{save_file_name}", "w") as f:
      print(pd_data, file=f)
    return pd_data.copy()

  def positions_changed_by_stochasticity(self):
    """
    This function assumes that nn architectures are ranked by their LHMDS.
    It then calculates the average change in position of an architecture when the LHMDS is influenced by
    stochasticity coming from different input data and different initialization.
    I simulate these by pertrubing lhmds of sampled nn architectures by standard deviation observed for
    these stochasticity sources.
    Experiments hinted that in both cases std is same across all models (e.g. if different init is the source of
    stochasticity all models' std is approx., 24)
    These STDs were 24 for different initializations and 4 for different data inputs used for lhmds calculation.
    I do 2 simulations:
      - First simulation assumes that stochasticity coming from images correlates amongst nn architectures
        e.g. if set of images 200-300 increase LHMDS of nn A, it will also increase LHMDS of network B by the same margin
        In this case I change lhmds only by +- 24
      - Second simulation assumes that images can have completly random influences on nn architectures
        e.g. if set of images 200-300 increase LHMDS of nn A, it doesn't tell us anything about its effect on network B
        In this case I change lhmds by +-28
    :return: average change position in first simulation(%), average change in position in first simulation(%),
             average change position in second simulation(raw), average change in position in second simulation(raw),
    """
    avg_pos_change_upper_bound = self.simulate_position_change(28, 800)
    avg_pos_change_lower_bound = self.simulate_position_change(24, 800)
    avg_pos_change_upper_percentage = avg_pos_change_upper_bound / len(self.experiment_data.experiment_list)
    avg_pos_change_lower_percentage = avg_pos_change_lower_bound / len(self.experiment_data.experiment_list)
    print_str = (f"Positions changed by log determinant higher estimate: {avg_pos_change_upper_bound}\n" 
                 f"Positions changed by log determinant lower estimate: {avg_pos_change_lower_bound}\n" 
                 f"Positions changed by log determinant higher estimate (%): {avg_pos_change_upper_percentage}\n" 
                 f"Positions changed by log determinant lower estimate (%): {avg_pos_change_lower_percentage}")
    with open(f"plots/{self.name}_statistics.txt", "a") as f:
      print(print_str, file=f)
    print(print_str)

    return (avg_pos_change_lower_percentage,
            avg_pos_change_upper_percentage,
            avg_pos_change_lower_bound,
            avg_pos_change_upper_bound)

  def simulate_position_change(self,
                               change_in_lhmds: float,
                               nn_architecture_samples: int) -> np.ndarray:
    """
    Simulates how much would change in lhmds by change_in_lhmds value change the rank of nn_architecture on average.
    :param change_in_lhmds: How much we should pertrube lhmds value
    :param nn_architecture_samples: How many architectures' LHMDS we should try to pertrube and average
    :returns: difference in positions if we added and substracted change_in_lhmds value.
    """
    self.experiment_data.experiment_list.sort(key=lambda x: x.log_determinant, reverse=False)
    log_determinants = [x.log_determinant for x in self.experiment_data.experiment_list]
    if nn_architecture_samples > len(self.experiment_data.experiment_list):
      nn_architecture_samples = len(self.experiment_data.experiment_list)

    indexes = np.random.choice(range(0, len(self.experiment_data.experiment_list)),
                               nn_architecture_samples,
                               replace=False)
    log_determinants_change_upwards = [log_determinants[i]+change_in_lhmds for i in indexes]
    log_determinants_change_downwards = [log_determinants[i]-change_in_lhmds for i in indexes]
    indexes_upper = np.array([bisect.bisect_left(log_determinants, x) for x in log_determinants_change_upwards])
    indexes_lower = np.array([bisect.bisect_right(log_determinants, x) for x in log_determinants_change_downwards])
    index_difference = indexes_upper - indexes_lower
    return index_difference

  def boxplot_lhmds(self,
                    title: str,
                    save_name: str,
                    x: str = "best_final_accuracy",
                    y: str = "log_determinant",
                    x_label: str = "LHMDS",
                    y_label: str = "Models"):
    """
    This function plots a boxplot of the log determinant by model.
    I used this to examine how much stochasticity influences the log determinant.
    :param title:
    :param save_name:
    :param x:
    :param y:
    :param x_label:
    :param y_label:
    :return:
    """
    data = self.experiment_data.to_pandas()
    data["best_final_accuracy"] = data["best_final_accuracy"].apply(lambda x: round(x, 2))
    data.sort_values(by="best_final_accuracy", ascending=True, inplace=True)
    plt.figure(figsize=(10, 10))
    plt.title(title, fontsize=35)
    ax = plt.subplot(1, 1, 1)
    sns.boxplot(x=x, y=y, data=data, ax=ax)
    ax.set_ylabel(x_label, fontsize=23)
    ax.set_xlabel(y_label, fontsize=23)

    ax.set_ylim(1100, 1450)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.savefig(f"plots/{save_name}.png")
    plt.show()