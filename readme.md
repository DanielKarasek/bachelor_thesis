# Contents
What is inside project can be read in my bachelor's thesis.

## How to run
Install python3, then install everything listed in requirements.txt.

<strong>!WARNING</strong> Possible there is some error with numpy going from np.bool to bool notation you have to fix these manually.

After that you will have to download datasets from [nasbench-101](https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord) and [nasbench-201](https://drive.google.com/open?id=1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs).
You have to save these files into this project in following locations and names: 
- nasbench/nasbench101/nasbench_only108.tfrecord
- nasbench/nasbench201/nasbench201.pth

## Example runs
 
In main loop of module entry.py, there are several methods you can run.
Each method has its docstring describing what it does and some other info.
Choose methods you want to run and run them.
You can also run all methods at once, but it will take a lot of time (and it might not work perfectly, since experiments
take too long I am unable to test it now myself :().

## brief overview of folders
- `data` - contains all processed data (e.g. sampled networks, experiment results etc.)
- `plots` - contains all plots generated by experiments
- `logs` - logs from some NN runs, I think it's not important, buty I am scared to delete it without testing everything :(
- `nasbench` - This folder contains modified nasbench-101 and nasbench-201 libraries and infrastracture used
to build KERAS models isomorphic to models described in nasbench papers.
  - `nasbench101` my modified version of nasbench-101 [git link](https://github.com/google-research/nasbench)
  - `nasbench201` my modified version of nasbench-201 [git link](https://github.com/D-X-Y/NAS-Bench-201)
- `nas_searchspaces` - 
  - `nassearchspaaces.py` - searchspaces (wrappers around nasbenches with common interface) used in experiments
  - `subsampling_interface.py` - methods for subsampling searchspaces
- `NASwithouttraining` - Contains code for NAS without training experiments
  - `stat_decorator_model.py` - Class which allows me to decorate NAS model in order to calculate LHDMS
  - `experiment_classes.py` - There are 2 types of experiment classes. First are experiment exucutors - they setup
                              experiment and run it. Second are experiment analysers - they analyse results of experiments
                              This folder also contains helper functions for dataset setup and dataclasses storing experiment results
- `MainFolder` 
  - `entry.py` - main file, contains high level methods for running experiments
  - `feature_extraction.py` - contains methods for extracting features from nasbench networks for regression methods
  - `graph_ops.py` - methods for analysis of DAG created from cell structure
  - `regression_experiments` - Experiments with regression methods (executor, analyser and dataclasses, this should be in separate folder)
  - `readme.md` - File with recursion in it (MUEHUEHUEHUEHEH 3:D)

## Known buggs
- During installation some numpy dependencies might still use np.\[type\], which included numpy version doesn't support
You have to change it manually in files, which are causing errors. (I am sorry about this, but we have to wait till all
libraries update their dependencies)
- Some examples in entry.py might not work perfectly (especially the long ones since I was unable to test them after 
                                                      interface updates because of computation time)
- During experiments keras throws some warnings, that suggest that I incorrectly built graph mode 
 (these warning shouldn't affect results, but might cause slower experiments).
