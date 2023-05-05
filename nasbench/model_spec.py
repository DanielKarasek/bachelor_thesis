# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model specification for module connectivity individuals.

This module handles pruning the unused parts of the computation graph but should
avoid creating any TensorFlow models (this is done inside model_builder.py).
"""


import copy

import numpy as np


class ModelSpec(object):
  """Model specification given adjacency original_matrix and labeling."""

  def __init__(self, original_matrix, ops, data_format='channels_last'):
    """Initialize the module spec.

    Args:
      original_matrix: ndarray or nested list with shape [V, V] for the adjacency matrix.
      ops: V-length list of labels for the base ops used. The first and last
        elements are ignored because they are the input and output vertices
        which have no operations. The elements are retained to keep consistent
        indexing.
      data_format: channels_last or channels_first.

    Raises:
      ValueError: invalid matrix or ops
    """
    if not isinstance(original_matrix, np.ndarray):
      original_matrix = np.array(original_matrix)
    shape = np.shape(original_matrix)
    if len(shape) != 2 or shape[0] != shape[1]:
      raise ValueError('original_matrix must be square')
    if shape[0] != len(ops):
      raise ValueError('length of ops must match original_matrix dimensions')
    if not self._is_upper_triangular(original_matrix):
      raise ValueError('original_matrix must be upper triangular')

    # Both the original and pruned matrices are deep copies of the original_matrix and
    # ops so any changes to those after initialization are not recognized by the
    # spec.

    self.matrix = copy.deepcopy(original_matrix)
    self.ops = copy.deepcopy(ops)
    self.valid_spec = True
    self._prune(original_matrix)

    self.data_format = data_format

  def _prune(self, original_matrix):
    """Prune the extraneous parts of the graph.

    General procedure:
      1) Remove parts of graph not connected to input.
      2) Remove parts of graph not connected to output.
      3) Reorder the vertices so that they are consecutive after steps 1 and 2.

    These 3 steps can be combined by deleting the rows and columns of the
    vertices that are not reachable from both the input and output (in reverse).
    """
    num_vertices = np.shape(original_matrix)[0]

    connected2input = self._get_connected_to_input(original_matrix)
    connected2output = self._get_connected_to_output(original_matrix)

    # Any vertex that isn't connected to both input and output is extraneous to
    # the computation graph.
    extraneous = set(range(num_vertices)).difference(
        connected2input.intersection(connected2output))

    # If the non-extraneous graph is less than 2 vertices, the input is not
    # connected to the output and the spec is invalid.
    if len(extraneous) > num_vertices - 2:
      self.matrix = None
      self.ops = None
      self.valid_spec = False
      raise ValueError("Input isn't connected to output")

    self.matrix = np.delete(self.matrix, list(extraneous), axis=0)
    self.matrix = np.delete(self.matrix, list(extraneous), axis=1)
    for index in sorted(extraneous, reverse=True):
      del self.ops[index]

  @staticmethod
  def _is_upper_triangular(matrix):
    """True if original_matrix is 0 on diagonal and below."""
    for src in range(np.shape(matrix)[0]):
      for dst in range(0, src + 1):
        if matrix[src, dst] != 0:
          return False

    return True

  @staticmethod
  def _get_equally_distributed_channels(vertex_channels: np.ndarray,
                                        matrix: np.ndarray,
                                        output_channels: int) -> np.ndarray:
    """
    Partitions output_channels between edges incoming into concatenation operation equally (+- 1 for correction).
    :return: Channel count for different vertices
    """

    concatenated_in_edges_count = np.sum(matrix[1:, -1], axis=0)
    # dividing output channels between input edges, that are concatenated
    interior_channels = output_channels // concatenated_in_edges_count
    # How much correction we need to apply (So all of them sum to output_channels)
    correction = int(output_channels % concatenated_in_edges_count)

    connected2output = np.where(matrix[1:, -1])[0]
    vertex_channels_ops_only = vertex_channels[1:-1]

    correction_indices = connected2output[:correction]
    vertex_channels_ops_only[connected2output] = interior_channels
    vertex_channels_ops_only[correction_indices] += 1

    vertex_channels[1:-1] = vertex_channels_ops_only

    return vertex_channels

  @staticmethod
  def _backtrack_channels(vertex_channels: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Backtracks values of channels partitioned by _get_equally_distributed_channels into whole DAG.

    Each vertex channels is set to max of all possible destinations. This is calculated iteratively
    starting from end of the DAG.
    :return: Channel count for different vertices
    """
    num_vertices = len(vertex_channels)
    vertex_channels_wo_output = vertex_channels[:-1]
    for v in range(num_vertices-3, 0, -1):
      dst_vector = np.asarray(matrix[v, :-1], bool)
      dst_vector[v] = True
      vertex_channels[v] = np.max(vertex_channels_wo_output[dst_vector])
      assert vertex_channels[v] > 0
    return vertex_channels

  @staticmethod
  def _sanity_check(vertex_channels: np.ndarray, matrix: np.ndarray, output_channels: int) -> None:
    """
    Sanity check if all vertex channel add up correctly and if they never increase going backwards in graph.
    :return: None, everything is asserted
    """
    num_vertices = len(vertex_channels)
    final_fan_in = 0
    for v in range(1, num_vertices - 1):
      if matrix[v, num_vertices - 1]:
        final_fan_in += vertex_channels[v]
      for dst in range(v + 1, num_vertices - 1):
        if matrix[v, dst]:
          assert vertex_channels[v] >= vertex_channels[dst]
    assert final_fan_in == output_channels or num_vertices == 2

  @staticmethod
  def compute_vertex_channels(input_channels: int, output_channels: int, matrix: np.ndarray):
    """Computes the number of channels at every vertex.

    All sizes are 1/N, where N is count of vertices incoming to the output node graph. Since 1/N isn't always
    whole number we must account for correction. These 1/N are then propagaded upwards since everywhere else
    layers are combined by element-wise addition instead of concatenation.
    Args:
      input_channels: input channel count.
      output_channels: output channel count.
      matrix: adjacency matrix for the module.
    Returns:
      list of channel counts, in order of the vertices.
    """
    num_vertices = np.shape(matrix)[0]

    vertex_channels = np.zeros(num_vertices)
    vertex_channels[0] = input_channels
    vertex_channels[num_vertices - 1] = output_channels

    vertex_channels = ModelSpec._get_equally_distributed_channels(vertex_channels, matrix, output_channels)
    vertex_channels = ModelSpec._backtrack_channels(vertex_channels, matrix)
    ModelSpec._sanity_check(vertex_channels, matrix, output_channels)

    return vertex_channels

  def _get_connected_to_input(self, matrix: np.ndarray):
    num_vertices = matrix.shape[0]
    visited_from_input = {0}
    frontier = [0]
    while frontier:
      top = frontier.pop()
      for v in range(top + 1, num_vertices):
        if matrix[top, v] and v not in visited_from_input:
          visited_from_input.add(v)
          frontier.append(v)
    return visited_from_input

  def _get_connected_to_output(self, matrix: np.ndarray):
    num_vertices = matrix.shape[0]
    visited_from_output = {num_vertices - 1}
    frontier = [num_vertices - 1]
    while frontier:
      top = frontier.pop()
      for v in range(0, top):
        if matrix[v, top] and v not in visited_from_output:
          visited_from_output.add(v)
          frontier.append(v)
    return visited_from_output
