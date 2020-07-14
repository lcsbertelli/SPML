# SPML implementation with MNIST dataset 

In this directory, SPML is implemented with the MNIST dataset. 
We have given implementation of SPML+native TensorFlow and SPML+privacy property

## SPML+native TensorFlow - Training
To run SPML+native TensorFlow for training we can use the following command:
```
python3 mnist_nativeTensorFlow.py
```
This will give us the trained model which can be feed for the inference later.

## SPML+native TensorFlow - Inference
To run SPML+native TensorFlow for inference we can use the following command:
```
python3 mnist_inference_native.py
```
We need to provide a model file path for inference

## SPML+privacy - Training
To run SPML+privacy for training, we need to set different epsilon values. In the current repository, we have tested our system with epsilon value as 0.1, 1, 2, 4, 6, and 8.
For the ease of our experiments, I have created 8 files, however, these can be optimized surely.

```
python3 mnist_.1.py
```

```
python3 mnist_1.py
```

```
python3 mnist_2.py
```

```
python3 mnist_4.py
```

```
python3 mnist_6.py
```

```
python3 mnist_8.py
```


## SPML+privacy - Inference
To run SPML+privacy for inference, we need to provide a respective trained model.
For the ease of our experiments, I have created 8 files, however, these can be optimized surely.

```
python3 mnist_inference_dp.1.py
```

```
python3 mnist_inference_dp2.py
```

```
python3 mnist_inference_dp3.py
```

```
python3 mnist_inference_dp4.py
```

```
python3 mnist_inference_dp5.py
```

```
python3mnist_inference_dp6.py
```

