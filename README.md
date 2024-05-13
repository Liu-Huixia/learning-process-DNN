<h2 align="center">
Understanding the learning process in deep neural networks with Information Bottleneck
</h2>


## Introduction

This repository presents the implemetation of experiments in thesis [[Understanding the learning process in deep neural networks with Information Bottleneck](https://trepo.tuni.fi/login/)], which is an application of the Information Bottleneck (IB) principle in deep learning [1]. The IB principle suggests that the learning process of Deep Neural Networks (DNNs) can be analyzed by quantifying the mutual information (MI) between the layers, the input, and the target variables. Here, Here we demonstrate how this quantification process can be implemented in deep feedforward neural networks (D-FFNN) with classification tasks.

## Setup
Please start by installing Miniconda with python. Python 3.11.5 was used here.

Please read the requirements.txt before you run this code.

NOTE: if you are using a Mac with python 3.7 and your Keras is 2.3.1, you may need to update the python or packgaes.


## Overall guidance for this repository
**![](overall.png)**


## Detailed explanation

* `MNIST_SaveActivations.ipynb` is a jupyter notebook that trains on MNIST and saves activation outputs when run on test set inputs for each epoch.
* `SZT_SaveActivations.ipynb` is a jupyter notebook that recreates the network and data from https://github.com/ravidziv/IDNNs and saves activations, for each epoch for a single trial.

* `loggingreporter.py` is a python file to save the activation values in DNNs at the end of each epoch.
  
* `MNIST_ComputeMI.ipynb` is a jupyter notebook that loads the data files, computes MI values, and does the infoplane plots for data created using `MNIST_SaveActivations.ipynb`.
* `SZT_ComputeMI.ipynb` is a jupyter notebook that loads the data files created by `IBnet_SaveActivations.ipynb`, computes MI values, and does the infoplane plots.

* `simplebinmi.py` is a python file to compute the MI based on the binning-based estimator [2].
* `kde.py` is a python file to compute the MI based on the pairwise-diastance estimator [3]. 
* `matrixRenyi.py` is a python file to compute the MI based on the matrix-based estimator [4].

* `Plots.ipynb` is a jupyter notebook that plots all figures in this thesis.
* `utils.py` is a python file mainly used to load datasets. You can edit this part as needed.
* `demo.py` is a simple script showing how to compute MI between X and Y, where Y = f(X) + Noise.

More details can be found in the code and comments in the corresponding files. If you have any questions and would like advice, please contact me.

## Acknowledgement

- This code is partially based on [artemyk/ibsgd] (https://github.com/artemyk/ibsgd/tree/master).
- The package `matrixRenyi.py` referred the code from [SJYuCNEL] (https://github.com/SJYuCNEL/brain-and-Information-Bottleneck).

## References

[1] N. Tishby, F. C. Pereira, and W. Bialek, “The information bottleneck method,” in Proc. 37th Annual Allerton Conference on Communications, Control and Computing, 1999, pp. 368–377.

[2] Shwartz-Ziv, Ravid, and Naftali Tishby. "Opening the Black Box of Deep Neural Networks via Information." Information Flow in Deep Neural Networks (2022): 24.

[3] Kolchinsky, Artemy, and Brendan D. Tracey. "Estimating mixture entropy with pairwise distances." Entropy 19, no. 7 (2017): 361.

[4] Giraldo, Luis Gonzalo Sanchez, Murali Rao, and Jose C. Principe. "Measures of entropy from data using infinitely divisible kernels." IEEE Transactions on Information Theory 61, no. 1 (2014): 535-548.