#  Times-C

 Times-C is a GPU-accelerated  k-Shape time series clustering algorithm.

# Installation

### CUDA toolkit

To install CUDA toolkit please use [this link](https://developer.nvidia.com/cuda-downloads).

### Dataset

Datasets can be obtained from the following websites:

[Welcome to the UCR Time Series Classification/Clustering Page](https://www.cs.ucr.edu/~eamonn/time_series_data/)

[Time Series Classification Website](http://www.timeseriesclassification.com/dataset.php)

### Compilation

Compile the code using the following command:

```
cd V2
make
```

### Execution

Run the code using the following command:

```
./test -d devNum /path/to/dataset k flag
```

Description of the arguments:

- -d devnum: Specify the device the code runs on
- /path/to/dataset : The relative or absolute path to the file containing time series
- k : Number of clusters
- flag: Whether the first column of the specified dataset is a label. If the first column is not a label, set it to 1, otherwise set it to 0. The first column of UCR database is a label, while the first column of UCR and UAE databases is not a label
- Example: `./test -d 0 ../data/InsectSound/InsectSound_TRAIN 10 1`

### Compare with other algorithms

Install tslearn:

```
python -m pip install tslearn
```

Run the code using the following command:

```
cd test
```

```
python original_test.py /path/to/dataset k flag
```

```
python tslearn_test.py /path/to/dataset k flag
```

Description of the arguments:

- /path/to/dataset : The relative or absolute path to the file containing time series
- k : Number of clusters
- flag: Whether the first column of the specified dataset is a label. If the first column is not a label, set it to 1, otherwise set it to 0. The first column of UCR database is a label, while the first column of UCR and UAE databases is not a label

# Publication

#### Abstract

As a basic structure of time series analysis, time series clustering has been widely used in many fields, but due to the high heterogeneity of time series data, it can not be simply regarded as a high-dimensional vector. Two time series often need some alignment method in similarity comparison. However, alignment between sequences is often a time-consuming process. For example, when using dynamic time warping as a sequence alignment algorithm and if the length of time series is greater than 1000, a single iteration in the clustering process may take hundreds to tens of thousands of seconds, while the entire clustering cycle often requires dozens of iterations. So far, the state-of-art time series clustering algorithm k-Shape provides high accuracy and scalability for time series clustering, although the k-Shape algorithm has a relatively lower computation cost than other time series clustering algorithms. However, when facing large-scale data, it still needs a long time to execute. In this paper, we propose an efficient parallel k-Shape algorithm implementation named gTSC. It can significantly reduce the time cost of clustering large amounts of high-dimensional time series data. Experiments show that our parallel k-Shape algorithm achieves a speedup of 160x.

#### Citation

```
@article{cuKshape,
	author = {Xun Wang , Ruibao Song , Junmin Xiao , Tong Li, and Xueqi Li}, 
	title = {Accelerating k-Shape Time Series Clustering Algorithm Using GPU},
	journal = {},
	year = {2022},
}
```
