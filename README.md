# Understanding the learning process in deep neural networks with Information Bottleneck

## Introduction

This repository presents the application of the Information Bottleneck (IB) principle in deep learning. The IB principle suggests that the learning process of Deep Neural Networks (DNNs) can be analyzed by quantifying the mutual information (MI) between the layers, the input data, and the target variables. Here, Here we demonstrate how this quantification process can be implemented in deep feedforward neural networks (D-FFNN) with classification tasks.

## Overall framework
**![](overall.png)**

## Code for Thesis work

* `MNIST_SaveActivations.ipynb` is a jupyter notebook that trains on MNIST and saves (in a data directory) activations when run on test set inputs for each epoch.

* `MNIST_ComputeMI.ipynb` is a jupyter notebook that loads the data files, computes MI values, and does the infoplane plots for data created using `MNIST_SaveActivations.ipynb`.

* `SZT_SaveActivations.ipynb` is a jupyter notebook that recreates the network and data from https://github.com/ravidziv/IDNNs and saves activations, for each epoch for a single trial.

* `SZT_ComputeMI.ipynb` is a jupyter notebook that loads the data files created by `IBnet_SaveActivations.ipynb`, computes MI values, and does the infoplane plots.

* `Plots.ipynb` is a jupyter notebook that plots all figures in this thesis.

* `simplebinmi.py` is a python file to compute the MI based on the binning-based estimator.
* `kde.py` is a python file to compute the MI based on the pairwise-diastance estimator.
* `matrixRenyi.py` is a python file to compute the MI based on the matrix-based estimator.

* `demo.py` is a simple script showing how to compute MI between X and Y, where Y = f(X) + Noise.


## Acknowledgement

- This code is partially based on [artemyk/ibsgd] (https://github.com/artemyk/ibsgd/tree/master).
- The package `matrixRenyi.py` referred the code from [SJYuCNEL] (https://github.com/SJYuCNEL/brain-and-Information-Bottleneck).


