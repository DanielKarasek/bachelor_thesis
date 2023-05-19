from __future__ import annotations

from NASWithoutTraining.experiment_classes.experiment_executors.rank_accuracy_experiment import StatisticsNasExperiment
from NASWithoutTraining.experiment_classes.experiment_executors.variance_experiment import VarianceExperiment
from NASWithoutTraining.experiment_classes.experiment_analysers.rank_accuracy_analyser import LHMDSAccuracyExperimentsAnalyser
from NASWithoutTraining.experiment_classes.experiment_analysers.variance_analyser import LHMDSStochasticityExperimentsAnalyser
from NASWithoutTraining.experiment_classes.experiment_dataclass import NASStatisticExperimentData
from nas_searchspaces.subsampling_interface import SearchSpaceSubsamplingInterface
from nasbench.model_builder import NASModel, NASBench201Model
from nas_searchspaces.nassearchspace import NASBench101, NASBench101Sampled, NASBench201
from feature_extractor import FeatureExtractor
from regression_experiments import RegressionExperimentator


def subsample_nasbench_101(save_path: str):
  search_space = NASBench101()
  sampling_interface = SearchSpaceSubsamplingInterface(search_space)
  sampling_interface.fill_bins()
  sampling_interface.subsamble_space(2000)
  sampling_interface.save(save_path)


def subsample_nasbench_201(save_path: str):
    search_space = NASBench201()
    sampling_interface = SearchSpaceSubsamplingInterface(search_space)
    sampling_interface.fill_bins()
    sampling_interface.subsamble_space(30)
    sampling_interface.save(save_path)


def extract_features_from_search_space(search_space, path_to_save: str):
  FeatureExtractor.get_data_and_save(search_space, path_to_save)


def extract_featerus_from_nasbench_101(path_to_save: str = "", subsampled_path: str = ""):
  if not subsampled_path:
    subsampled_path = "data/nasbench-101-subsample"
  if not path_to_save:
    path_to_save = "data/features_from_nasbench_101.npz"
  search_space = NASBench101Sampled(SearchSpaceSubsamplingInterface
                                    .load(subsampled_path), NASModel)
  extract_features_from_search_space(search_space, path_to_save)


def extract_featerus_from_nasbench_201(path_to_save: str = "", subsampled_path: str = ""):
  if not subsampled_path:
    subsampled_path = "data/nasbench-201-subsample"
  if not path_to_save:
    path_to_save = "data/features_from_nasbench_201.npz"
  search_space = NASBench101Sampled(SearchSpaceSubsamplingInterface
                                    .load(subsampled_path), NASBench201Model)
  extract_features_from_search_space(search_space, path_to_save)


def run_nas_without_training_experiment(search_space, output_file: str = ""):
  experiment_wrapper = StatisticsNasExperiment(search_space)
  experiment_wrapper.run_experiment()
  if not output_file:
    output_file = "data/LHMDS-nasbench-101-experiments"
  experiment_wrapper.save_experiments(output_file)


def run_regressor_experiment(path_to_features: str, experiment_name: str = ""):
  experiment = RegressionExperimentator(path_to_features)
  experiment.run_experiment()
  with open(f"data/{experiment_name}_regression_experiment_results.txt", "w") as f:
    print(experiment.results, file=f)


def run_nasbench_201_variance_experiment(search_space):
  experiment = VarianceExperiment(search_space)
  experiment.run_experiment()
  experiment.save("data/variance_experiment_nasbench_201")


def LHMDS_statistic_experiments_example():
  """
  This method loads sampled subspaces from nasbench-101 and nasbench-201 and
  performs all 3 experiments from the paper.
  This is rather slow and there are already precalculated results in the data folder.
  So feel free to skip this method.
  :return:
  """
  nas_101_sampled_NNs = (SearchSpaceSubsamplingInterface
                         .load_sampled_experiments("data/nasbench-101-subsample"))
  nas_201_sampled_NNs = (SearchSpaceSubsamplingInterface
                         .load_sampled_experiments("data/nasbench-201-subsample"))
  # nas_201_sampled_NNs_variance = (SearchSpaceSubsamplingInterface
  #                                 .load_sampled_experiments("data/variance_test_subsample"))
  # nas_201_sampled_NNs_variance.save_sampled_experiments("data/variance_test_subsample")
  # nas_101_sampled_NNs.save_sampled_experiments("data/nasbench-101-subsample")
  # nas_201_sampled_NNs.save_sampled_experiments("data/nasbench-201-subsample")
  nas101_search_space = NASBench101Sampled(nas_101_sampled_NNs, NASBench201Model)
  nas201_search_space = NASBench101Sampled(nas_201_sampled_NNs, NASBench201Model)

  # nas201_search_space_variance = NASBench101Sampled(nas_201_sampled_NNs_variance,
  #                                                   NASBench201Model)

  # run_nasbench_201_variance_experiment(nas201_search_space_variance)
  run_nas_without_training_experiment(nas101_search_space, "data/LHMDS-nasbench-101-experiments")
  run_nas_without_training_experiment(nas201_search_space, "data/LHMDS-nasbench-201-experiments")


def subsampling_example():
  """
  This method subsamples nasbench-101 and nasbench-201 search spaces.
  NASbench-201 is super innefiecient and requieres over 20 GB ram
  (DAMN YOU RESEARCHERS WITH HIGH SPEC PCS)
  This method doesnt have to be run since results are already in the data folder.
  :return:
  """
  subsample_nasbench_101("data/nasbench-101-subsample")
  subsample_nasbench_201("data/nasbench-201-subsample")


def show_lhmds_results():
  """
  This example shows how you can load the experiments and plot them.
  There are more plotable experiments, so feel free to test plot functions
  Its a bit of a cluster fuck now, (lot of titles and stuff are hardcoded)
  """

  experiments = NASStatisticExperimentData.load("data/LHMDS-nasbench-201-experiments", "nasbench-201-LHMDS")

  analyser = LHMDSAccuracyExperimentsAnalyser(experiments)
  analyser.plot_inferred_accuracies_with_stats()

  experiments = NASStatisticExperimentData.load("data/nas201_different_inits.pickle")
  analyser = LHMDSStochasticityExperimentsAnalyser(experiments)
  analyser.boxplot_lhmds("Effect of different initializations", "different_initialization_nasbench_201")


def extract_features_and_run_regressors():
  """
  This method extract features both for nasbench-101 and nasbench-201 and runs regressors
  on them. Results are showed straight away (and saved in data folder)
  :return:
  """
  extract_featerus_from_nasbench_101(path_to_save="data/features_from_nasbench_101.npz",
                                     subsampled_path="data/nasbench-101-subsample")
  extract_featerus_from_nasbench_201(path_to_save="data/features_from_nasbench_101.npz",
                                     subsampled_path="data/nasbench-101-subsample")

  run_regressor_experiment("data/features_from_nasbench_101.npz", "nasbench-101")
  run_regressor_experiment("data/features_from_nasbench_201.npz", "nasbench-101")


def main():
  # subsampling_example()
  # LHMDS_statistic_experiments_example()
  show_lhmds_results()
  # extract_features_and_run_regressors()


if __name__ == "__main__":
  main()
