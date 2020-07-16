# SPML implementation with Randomized noise

In this directory, SPML is implemented with randomized noise. 
This technique is based on adding randomness to each person's response. The randomness is added by flipping a coin. For example, if simple questions are asked which can be answered in yes/no, then to answer these questions (1) A coined is flipped two times (2) If its head question is honestly answered (3) If its tail, then second coin flip comes into play and yes is answered for the head and no is answered for the tail. This provides plausible deniability because it cannot be traced if the answer is honestly answered or its due to the coin flip.

## Directory Structure
1. Training - It contains scripts to train the model.
2. Models - It contains an already trained model.
3. Inference - It contains scripts for inference an already or new trained model.
4. Logs - It contains execution or training logs from already trained models.

## SPML+native TensorFlow - Training
To run SPML+native TensorFlow for training we can use the following command:
```
python3 adultnonrr.py
```
This will give us the trained model which can be feed for the inference later.

## SPML+native TensorFlow - Inference
To run SPML+native TensorFlow for inference we can use the following command:
```
python3 adultnonrrinference.py
```
We need to provide a model file path for inference

## SPML+privacy - Training
To run SPML+privacy for training, we need to set different epsilon values. In the current repository, we have tested our system with epsilon value as 0.13, 0.53, 1, 2, 3, 4.
For the ease of our experiments, I have created 6 files, however, these can be optimized surely.

```
python3 adultrr_.13.py
```

```
python3 adultrr_.53.py
```

```
python3 adultrr_1.py
```

```
python3 adultrr_2.py
```

```
python3 adultrr_3.py
```

```
python3 adultrr_4.py
```


## SPML+privacy - Inference
To run SPML+privacy for inference, we need to provide a respective trained model.
For the ease of our experiments, I have created 6 files, however, these can be optimized surely.


```
python3 adultrrinference_.13.py
```

```
python3 adultrrinference_.53.py
```

```
python3 adultrrinference_1.py
```

```
python3 adultrrinference_2.py
```

```
python3 adultrrinference_3.py
```

```
python3 adultrrinference_4.py
```
