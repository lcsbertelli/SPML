# SPML: A Secure Privacy-preserving Machine Learning Framework using TEEs and Differential Privacy

This repository contains the implementation of the SPML. SPML system is developed 
in Python using TensorFlow and TensorFlow Privacy library

The privacy property is enabled via differential privacy technique and security property is enabled via TEEs.
SPML system is tested on Intel SGX as TEEs and integrated via SCONE library.

## Setting up SPML

### Dependencies

1. SPML system uses TensorFlow (version=1.15) hence TensorFlow should be installed in the system.
2. For privacy property to work TensorFlow privacy library should be installed.
3. Some other dependencies include NumPy, Pandas, Sklearn.

### Installing TensorFlow Privacy
Tensorflow privacy library can be installed using:

```
pip install tensorflow_privacy
```

## Keras
The current implementation of SPML runs with Keras version=2.3.1. Due to some open issues at the TensorFlow library, we have to do some patchwork in Keras library. Hence to train models with the Privacy library I have included the local copy of Keras version to reduce the system dependencies.

## Recommendation
Currently, the privacy library doesn't support TensorFlow 2.x hence SPML also doesn't support 2.x.
The current recommendation is to use TensorFlow version 1.15 with a local copy of Keras library.
