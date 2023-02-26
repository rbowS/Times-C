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

Time-series analysis has appeared in many field, such as biology, astronomy, meteorology, finance, and others. Clustering is one of the most critical methods in time-series analysis. So far, the state-of-art time series clustering algorithm k-Shape has been widely used not only because of its high accuracy, but also because of its relatively low computation cost. However, due to the high heterogeneity of time series data, it can not be simply regarded as a high-dimensional vector. Two time series often need some alignment method in similarity comparison. The alignment between sequences is often a time-consuming process. For example, when using dynamic time warping as a sequence alignment algorithm and if the length of time series is greater than 1000, a single iteration in the clustering process may take hundreds to tens of thousands of seconds, while the entire clustering cycle often requires dozens of iterations. In this paper, we propose three new parallel methods for aggregation, centroid, and class assignment, called Times-C, which is an efficient parallel k-Shape algorithm design and implementation . Overall, experiments show that Times-C achieves a 198*×* speedup compared to the out-of-order processor.

#### Citation

```
@article{cuKshape,
	author = {Xun Wang , Ruibao Song , Junmin Xiao , Tong Li, and Xueqi Li}, 
	title = {Accelerating k-Shape Time Series Clustering Algorithm Using GPU},
	journal = {},
	year = {2022},
}
```
