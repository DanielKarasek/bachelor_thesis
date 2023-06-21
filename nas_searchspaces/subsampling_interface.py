from __future__ import annotations

import pickle
import numpy as np

<<<<<<< HEAD
from nas_searchspaces.nassearchspace import (NASBench101)
=======
from nas_searchspaces.nassearchspaces import (NASBench101)
>>>>>>> 9065cc3cb3fc80a960d72274d2dc4fc463996d31
from nasbench.model_spec import ModelSpec

class SearchSpaceSubsamplingInterface:
  """
  Class used for subsampling search_space, while keeping diversity in final
  validation of sampled networks
  """

  def __init__(self, searchspace: NASBench101):
    self.bins = [0.0, 50.0, 70.0, 80.0, 85.0, 87.0, 89.0, 90.0, 90.8,
                 91.5, 92.0, 92.5, 93.0, 94.0, 94.5, 95.0, 100.0]
    self.binned_hashes = [[] for _ in range(len(self.bins) - 1)]
    self.searchspace = searchspace
    self.sampled_experiments = []

  def find_bin_idx(self, final_accuracy: float):
    # first index bigger than final_accuracy
    return next(idx - 1 for idx in range(len(self.bins)) if self.bins[idx] > (final_accuracy * 100))

  def fill_bins(self) -> None:
    for unique_hash in self.searchspace:
<<<<<<< HEAD
      final_accuracy = self.searchspace.get_final_accuracy(unique_hash)
      model_spec: ModelSpec = self.searchspace.get_spec(unique_hash)
      bin_idx = self.find_bin_idx(final_accuracy)
      self.binned_hashes[bin_idx].append([unique_hash, final_accuracy, model_spec.matrix, model_spec.ops])

=======
      try:
        final_accuracy = self.searchspace.get_final_accuracy(unique_hash)
        model_spec: ModelSpec = self.searchspace.get_spec(unique_hash)
        bin_idx = self.find_bin_idx(final_accuracy)
        self.binned_hashes[bin_idx].append([unique_hash, final_accuracy, model_spec.matrix, model_spec.ops])
      except ValueError as e:
        print(e.__str__())
>>>>>>> 9065cc3cb3fc80a960d72274d2dc4fc463996d31
  def subsamble_space(self, total_samples: int = 1000):
    percentages = np.array([0.02, 0.03, 0.04, 0.05, 0.06,
                            0.07, 0.08, 0.11, 0.11, 0.12,
                            0.12, 0.11, 0.11, 0.06])
    samples_per_bin = np.asarray(percentages * total_samples, np.int32)
    self.sampled_experiments = []
    sampled_experiments = []
    for bin_idx, single_bin in enumerate(self.binned_hashes[:14]):
      sample_count = min(samples_per_bin[bin_idx], len(single_bin))
      random_choice_indexes = np.random.choice(np.arange(len(single_bin)), size=sample_count, replace=False)
      sampled_experiments.append([single_bin[idx] for idx in random_choice_indexes])

    for sampled_sublist in sampled_experiments:
      for sampled_experiment in sampled_sublist:
        self.sampled_experiments.append(sampled_experiment)

  def save_sampled_experiments(self, path):
    with open(path, "wb") as output_file:
      pickle.dump(self.sampled_experiments, output_file)

  @staticmethod
  def load_sampled_experiments(path):
    with open(path, "rb") as input_file:
      return pickle.load(input_file)

  def save(self, path):
    with open(path, "wb") as output_file:
      pickle.dump(self, output_file)

  @staticmethod
  def load(path) -> SearchSpaceSubsamplingInterface:
    with open(path, "rb") as input_file:
      return pickle.load(input_file)
