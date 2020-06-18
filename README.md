# MPI_CovarianceMatrix
To calculate covariance matrix of the given dataset by MPI
iris.data is given as a dataset. Read the data from file and ignore the last column of each row in dataset.

To compile:
```
mpicc -o cov cov.c
```
where cov is the name of the object file.

To run:
```
mpiexec -n 4 ./cov
```
where 4 is the number of processors. You can give any positive integer value to n. But generally machines have 4 processors, so giving value of n greater 4 will create lot of overhead and actually increase the time taken.
