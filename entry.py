from __future__ import annotations

import logging

import tensorflow.python.keras as keras

from NASWithoutTraining.stat_decorator_model import StatDecoratorModel
from NASWithoutTraining.experiment import StatisticsNasExperiment
from nas_searchspaces.subsampling_interface import SearchSpaceSubsamplingInterface
from nasbench.constants import Operations
from nasbench.model_builder import NASModel
from nasbench.model_spec import ModelSpec
from nas_searchspaces.nassearchspace import NASBench101, NASBench101Sampled
from feature_extractor import FeatureExtractor
from regression_experiments import RegressionExperimentator

from nasbench.nasbench201.nasbench_201_api import NASBench201API


def subsample_nasbench_101(save_path: str):
  search_space = NASBench101()
  sampling_interface = SearchSpaceSubsamplingInterface(search_space)
  sampling_interface.fill_bins()
  sampling_interface.subsamble_space(2000)
  sampling_interface.save(save_path)


def subsample_nasbench_201(save_path: str):
  pass


def extract_features_from_search_space(search_space, path_to_save: str):
  FeatureExtractor.get_data_and_save(search_space, path_to_save)


def extract_featerus_from_nasbench_101():
  search_space = NASBench101Sampled(SearchSpaceSubsamplingInterface
                                    .load("./sampled_experiments")
                                    .sampled_experiments)
  extract_features_from_search_space(search_space, "./features_from_nasbench_101.npz")


def run_nas_without_training_experiment(search_space):
  experiment_wrapper = StatisticsNasExperiment(search_space)
  experiment_wrapper.run_experiment()
  experiment_wrapper.save_experiments("./NASWithoutTrainingExperiment")


def run_regressor_experiment(path_to_features: str):
  experiment = RegressionExperimentator(path_to_features)
  experiment.run_experiment()
  print(experiment.results)


def main():
  logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
  # extract_featerus_from_nasbench_101()
  run_regressor_experiment("./features_from_nasbench_101.npz")
  # space = NASBench101Sampled(SearchSpaceSubsamplingInterface.load("./sampled_experiments").sampled_experiments)
  # run_nas_without_training_experiment(search_space=space)


  experiment = StatisticsNasExperiment.load("./NASWithoutTrainingExperiment")
  print(experiment.experiments.calculate_r_squared())
  print(experiment.experiments.calculate_kendall_tau_statistics())
  experiment.experiments.plot_infered_accuracies()


if __name__ == "__main__":
  main()


def iterate_over_sampled_101_bench():
  sampled_experiments = SearchSpaceSubsamplingInterface.load_sampled_experiments("./sampled_experiments")
  search_space = NASBench101Sampled(sampled_experiments)
  for idx in search_space:
    print(search_space.get_final_accuracy(idx))


def setup_example_model_spec():
  model_spec = ModelSpec(
    # Adjacency matrix of the module
    original_matrix=[[0, 1, 0, 1, 0, 1, 1],
                     [0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0]],  # output layer
    # Operations at the vertices of the module, matches order of matrix
    ops=[Operations.INPUT.value, Operations.CONV1X1.value,
         Operations.CONV3X3.value, Operations.CONV3X3.value,
         Operations.CONV3X3.value, Operations.MAXPOOL3X3.value, Operations.OUTPUT.value])
  nas_model = NASModel(model_spec)
  keras_inputs = keras.layers.Input((28, 28, 1), 64)
  first_module = StatDecoratorModel(nas_model)(keras_inputs)
  model = keras.Model(inputs=keras_inputs, outputs=first_module)
  model.compile(loss=keras.losses.categorical_crossentropy)
  return
