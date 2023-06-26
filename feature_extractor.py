from typing import List

import numpy as np

import graph_ops
from nasbenches.constants import Operations
from nasbenches.model_spec import ModelSpec


class FeatureExtractor:

  @staticmethod
  def get_features_from_cell(all_cell_specs: List[ModelSpec]):
    g_widths = [graph_ops.get_total_paths_from_adjacency_matrix(cell_spec.matrix) for cell_spec in all_cell_specs]
    g_depths = [graph_ops.get_longest_path_from_adjacency_matrix(cell_spec.matrix) for cell_spec in all_cell_specs]
    operation_values = [op.value for op in Operations if op.value != "input" or op.value != "output"]
    g_op_counts = [[cell_spec.ops.count(op) for op in operation_values] for cell_spec in all_cell_specs]
    for idx, single_op_count in enumerate(g_op_counts):
      single_op_count.append(g_widths[idx])
      single_op_count.append(g_depths[idx])
    return np.array(g_op_counts)

  @staticmethod
  def get_data_from_search_space(search_space):
    final_accuaracies: List[float] = [search_space.get_final_accuracy(unique_hash) for unique_hash in search_space]
    all_model_specs: List[ModelSpec] = [search_space.get_spec(unique_hash) for unique_hash in search_space]
    x_train = FeatureExtractor.get_features_from_cell(all_model_specs)
    y_train = np.array(final_accuaracies)
    return x_train, y_train

  @staticmethod
  def get_data_and_save(search_space, path):
    x_train, y_train = FeatureExtractor.get_data_from_search_space(search_space)
    np.savez(path, x_train=x_train, y_train=y_train)