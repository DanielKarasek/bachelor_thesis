from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import List

import pandas as pd

import sys


@dataclass
class StatisticNASSingleEntryResult:
  """Basic class for storing results of a single experiment."""
  __slots__ = ['unique_hash', 'log_determinant', 'best_final_accuracy'] # log_determinant is the LHMDS statistic
  unique_hash: str  # unique hash of the architecture
  log_determinant: float  # LHMDS statistic
  best_final_accuracy: float  # best final accuracy of the architecture

  def to_list(self):
    return [self.unique_hash, self.log_determinant, self.best_final_accuracy]


@dataclass
class NASStatisticExperimentData:
  """Basic class for storing results set of experiments."""
  __slots__ = ['experiment_list', 'name']
  experiment_list: List[StatisticNASSingleEntryResult]
  name: str

  def add_experiment(self, experiment: StatisticNASSingleEntryResult):
    self.experiment_list.append(experiment)

  def save(self, path):
    with open(path, "wb") as output_file:
      pickle.dump(self, output_file)

  @staticmethod
  def load(path, name="") -> NASStatisticExperimentData:
    with open(path, "rb") as input_file:

      me = pickle.load(input_file)
      me.name = name
      me.save(path)
      return me

  def to_pandas(self):
    return pd.DataFrame([x.to_list() for x in self.experiment_list], columns=["unique_hash",
                                                                              "log_determinant",
                                                                              "best_final_accuracy"])

  def sort_by_final_accuracy(self):
    self.experiment_list.sort(key=lambda x: x.best_final_accuracy, reverse=True)
