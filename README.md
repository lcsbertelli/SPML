# SPML:A Secure Privacy preserving Machine Learning Framework using TEEs and Differential Privacy

This repository contains the implementation of the SPML. SPML system is developed 
in Python using TensorFlow and TensorFlow Privacy library

The privacy property is enabled via differential privacy techqniue and security property is enabled via TEEs.
SPML system is tested on Intel SGX as TEEs and integrated via SCONE library.

## Setting up SPML

### Dependencies

1. SPML system uses TensorFlow (version=1.15) hence tensorflow should be installed in the system.
2. For privacy property to work TensorFlow privacy library should be installed.
3. Some other dependecies include numpy, pandas, sklearn.

### Installing TensorFlow Privacy
Tensorflow privacy library can be installed using:

```
pip install tensorflow_privacy
```

## SPML directory
In SPML directory three directories are available
1. mnist - SPML implementation with mnist dataset 
2. cifar10 - SPML implementation with cifar10 dataset 
3. rr - SPML implementation with randomized response 


# SPML
# SPML
