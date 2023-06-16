### EPFL 2022-2023: Machine Learning Project 1 - Finding the Higgs Boson

This repository contains the following :

* An `implementations.py` file for part 1
* A `run.py` script which produces our predictions
* A report describing our work

In the `implementations.py` file, we implemented the six regression models :
Gradient Descent, Stochastic Gradient Descent, Least Squares, Ridge Regression, Logistic Regression and finally,
Regularised Logistic Regression.

We also provided :

* helpers.py : contains useful function such as loss functions or cross validation function
* HW1.ipynb : the whole process (data cleaning, preprocessing, training) was done on this notebook
* run.py : file to run that should give the predictions

## Running the code

To make the code work out of the box, the datasets must be put in `resources`, ie have `resources/train.csv` and
`resources/test.csv`.

The code execute itself by launching `python3 run.py`. The inputs are in the data folder. We used a cross validation
with a ridge regression on the training dataset. There should be no problem for the execution of the file coded in
Python 3. The only library used is `numpy`. 

