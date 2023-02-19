### Compilation

Compile the code using the following command:

```
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



