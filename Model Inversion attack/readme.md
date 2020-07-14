# SPML - Model Inversion Attack

The Model Inversion Attack was demonstrated by Matthew Fredrikson (https://www.usenix.org/conference/usenixsecurity14/technical-sessions/presentation/fredrikson_matthew)

The model inversion attack states, if an adversary has access to a trained model and some auxiliary information about an individual, then it can make predictions about an individual’s genetic markers. This particular attack is discussed via an example from the pharmacogenetics field in the paper, but it can be generalized in any scenario like we have used this type of attack on the classification of hand-written digits from MNIST dataset.

## Implementation of the Attack
The attack is implemented with the help of IBM Adversarial Robustness Toolbox (ART) library. (https://github.com/Trusted-AI/adversarial-robustness-toolbox)
ART is Python library which help us to evaluate and verify our Machine learning applications. It supports various types of attacks on machine learning such as Inference, Evasion, Poisoning and Extraction. For our work, we are interested in Model Inversion attack, hence we implemented this attack with help of Python APIs available as part of this library. 

## Prerequisite
We need ART library to implement the attack and it can be installed as:

```
pip install adversarial-robustness-toolbox
```

## How to run
The Model inversion attack is a black-box type attack over the machine learning model, hence to demonstrate the attack, we need a trained models with good accuracy. In the model directory, various trained models are already kept for MNIST dataset. The trained model includes model trained with native TensorFlow and privacy library with different epsilon value.
```
python3 mnist_attack_full_native.py
```
We need to provide model file path for attack
