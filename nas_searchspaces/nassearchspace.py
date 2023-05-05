from nasbench.nasbench101.nasbench_101_api import NASBench
from nasbench.model_spec import ModelSpec
from nasbench.model_builder import NASModel, NASBench201Model

from nasbench.nasbench201.nasbench_201_api import NASBench201API
import tensorflow.python.keras as keras


class NASBench201:
  def __init__(self, tfrecord_location: str = "./nasbench/nasbench201/nasbench201.pth"):
    self.nas = NASBench201API(tfrecord_location)

  def get_spec(self, idx: int) -> ModelSpec:
    matrix, ops, _ = self.nas.get_more_info(idx, "cifar10", hp=200, is_random=True)
    return ModelSpec(matrix, ops)

  def get_network(self, idx: int) -> keras.Model:
    cell_spec = self.get_spec(idx)
    nas_model = NASBench201Model(cell_spec)
    return nas_model

  def get_final_accuracy(self, idx: int) -> float:
    return self.nas.get_more_info(idx, "cifar10", hp=200, is_random=True)[2]['test-accuracy']/100

  def __iter__(self):
    for idx in self.nas:
      yield idx


class NASBench101:
  def __init__(self, tfrecord_location: str = "./nasbench/nasbench101/nasbench_only108.tfrecord"):
    self.nas = NASBench(tfrecord_location, 1000)

  def get_spec(self, unique_hash: str) -> ModelSpec:
    matrix = self.nas.fixed_statistics[unique_hash]['module_adjacency']
    operations = self.nas.fixed_statistics[unique_hash]["module_operations"]
    return ModelSpec(matrix, operations)

  def get_network(self, unique_hash: str) -> keras.Model:
    cell_spec = self.get_spec(unique_hash)
    nas_model = NASModel(cell_spec)
    return nas_model

  def get_final_accuracy(self, unique_hash: str) -> float:
    sum_accuracy = 0
    single_spec_results = self.nas.computed_statistics[unique_hash][108]
    for exp_num, experiment in enumerate(single_spec_results, 1):
      experiment_test_accuracy = experiment["final_test_accuracy"]
      sum_accuracy += experiment_test_accuracy
    return sum_accuracy/exp_num

  def __iter__(self):
    for unique_hash in self.nas.hash_iterator():
      yield unique_hash


class NASBench101Sampled:
  """
  Same as nasbench101 but uses just subsample of all networks,
  created by SearchSpaceSubsamplingInterface class
  """
  def __init__(self, experiment_list, model_class):
    self.model_class = model_class
    self.experiments = experiment_list

  def get_spec(self, idx: int) -> ModelSpec:
    matrix = self.experiments[idx][2]
    operations = self.experiments[idx][3]
    return ModelSpec(matrix, operations)

  def get_network(self, idx: int) -> keras.Model:
    cell_spec = self.get_spec(idx)
    nas_model = self.model_class(cell_spec)
    return nas_model

  def get_final_accuracy(self, idx: int) -> float:
    return self.experiments[idx][1]

  def __iter__(self):
    for idx in range(len(self.experiments)):
      yield idx

