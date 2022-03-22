# Guided-Adversarial-Augmentation
Code and Dataset for the ACL 2022 (Findings) paper "Leveraging Expert Guided Adversarial Augmentation For Improving Generalization in Named Entity Recognition"

arXiv link: https://arxiv.org/abs/2203.10693

## Requirements
Python 3.6 or higher

Pytorch >= 1.4.0

Pytorch_transformers (also known as transformers)

Numpy

## Data
Please unzip the data.zip file. It contains within it the Challenge Set. The Challenge Set’s examples are annotated with a 1 for high quality and a 0 or 2 for low
quality. The code only reads in examples annotated with a 1 from the file.

## Code
Please run the commands contained in the "README Commands" file for data processing and reproduction of experiments.
