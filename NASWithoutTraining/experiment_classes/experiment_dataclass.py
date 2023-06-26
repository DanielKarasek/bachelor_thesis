from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import List

import pandas as pd

import sys


@dataclass
class StatisticNASSingleEntryResult:
  """Basic class for storing results of a single experiment."""
  __slots__ = ['unique_hash', 'log_determinant', 'best_final_accuracy', 'repeat_id'] # log_determinant is the LHMDS statistic
  unique_hash: str  # unique hash of the architecture
  log_determinant: float  # LHMDS statistic
  best_final_accuracy: float  # best final accuracy of the architecture
  repeat_id: int   # id of the repeat

  def to_list(self):
    return [self.unique_hash, self.log_determinant, self.best_final_accuracy, self.repeat_id]


@dataclass
class NASStatisticExperimentData:
  """Basic class for storing results set of experiments."""
  __slots__ = ['experiment_list', 'name', 'version']
  experiment_list: List[StatisticNASSingleEntryResult]
  name: str
  version: str

  def __init__(self, experiment_list=None, name="", version="1.1"):
    self.experiment_list = []
    self.name = name
    self.version = version

  def add_experiment(self, experiment: StatisticNASSingleEntryResult):
    self.experiment_list.append(experiment)

  def save(self, path):
    with open(path, "wb") as output_file:
      pickle.dump(self, output_file)

  @staticmethod
  def load(path, name="") -> NASStatisticExperimentData:
    with open(path, "rb") as input_file:
      me = pickle.load(input_file)
      hasattr(me, "version") or setattr(me, "version", "1.0")
      if me.version == "1.0":
        for experiment in me.experiment_list:
          experiment.repeat_id = 0
      me.name = name
      return me

  def to_pandas(self):
    return pd.DataFrame([x.to_list() for x in self.experiment_list], columns=["unique_hash",
                                                                              "log_determinant",
                                                                              "best_final_accuracy",
                                                                              "repeat_id"])

  def sort_by_final_accuracy(self):
    self.experiment_list.sort(key=lambda x: x.best_final_accuracy, reverse=True)
