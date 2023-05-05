# Contains
What is inside project can be read in my bachelor's thesis.

# How to run
Install python3, then install everything listed in requirements.txt.
Possible there is some error with numpy going from np.bool to bool notation
you have to fix this manually.
After that you will have to download datasets from [nasbench-101](https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord) and [nasbench-201](https://drive.google.com/open?id=1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs).
You have to save this files into this project as 
- nasbench/nasbench101/nasbench_only108.tfrecord and 
- nasbench/nasbench201/nasbench201.pth


 
In main loop of module entry.py, there are several methods you can run.
Each method has its docstring describing what it does and some other info.
Choose methods you want to run and run them.
You can also run all methods at once, but it will take a lot of time.

# brief overview of folders
- `data` - contains data processed data (e.g. sampled networks, experiment results etc.)
- 'nasbench' - nasbench libraries from their official github[nas101](https://github.com/google-research/nasbench)[nas201](https://github.com/D-X-Y/NAS-Bench-201)
  you have to use main version of these libraries, since I modified them quite a bit. It also contains my model building infrastracture
- nas searchspaces - wrapper allowing creating nasbench searchspaces with same interface for different nasbenches
- NASwithouttraining - Contains code for NAS without training experiments
- MainFolder 
  - entry.py - main file, contains high level methods for running experiments
  - feature_extraction.py - contains methods for extracting features from nasbench networks for regression methods
  - graph_ops.py - methods for analysis of cell DAGs
  - regression_experiments - Experiments with regression methods
  - 