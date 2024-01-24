# Revisiting and Advancing Privacy Attribute Preserving from an Attack Perspective

This project is for the paper: Revisiting and Advancing Privacy Attribute Preserving from an Attack Perspective.

The code was developed on Python 3.6.13 and Tensorflow 1.14.0

## Usage

### 1. Attribute inference attack without any protection measures
```
usage: python main.py --num 0 --encrypt True --test True [--model MODEL] [--gpu GPU_ID] [--dataset DATASET_NAME]

arguments:
  --model MODEL
                        support: logist, dnn, svd, rf
  --gpu GPU_ID
                        GPU ID, default is 0.
  --dataset DATASET_NAME
                        support: app, weibo.
```

### 2. Obtain the perturbation version of the public users
```
usage: python main.py --encrypt True [--num NUM] [--gpu GPU_ID] [--dataset DATASET_NAME]

arguments:
  --num NUM
                        perturbation budget
  --gpu GPU_ID
                        GPU ID, default is 0.
  --dataset DATASET_NAME
                        support: app, weibo.
```

### 3. Evaluating the privacy-preserving performance of perturbed data
```
usage: python main.py --encrypt True --test True [--num NUM] [--gpu GPU_ID] [--dataset DATASET_NAME]

arguments:
  --num NUM
                        perturbation budget
  --gpu GPU_ID
                        GPU ID, default is 0.
  --dataset DATASET_NAME
                        support: app, weibo.
```