# Guided-Adversarial-Augmentation

Code and Dataset for the paper:

*Aaron Reich, Jiaao Chen, Aastha Agrawal, Yanzhe Zhang, Diyi Yang*: Leveraging Expert Guided Adversarial Augmentation to Improve Generalization in Named Entity Recognition, ACL 2022 (Findings).

If you would like to refer to it, please cite the paper mentioned above ([Arxiv](https://arxiv.org/abs/2203.10693)).

```
@misc{reich2022leveraging,
    title={Leveraging Expert Guided Adversarial Augmentation For Improving Generalization in Named Entity Recognition},
    author={Aaron Reich and Jiaao Chen and Aastha Agrawal and Yanzhe Zhang and Diyi Yang},
    year={2022},
    eprint={2203.10693},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Requirements
python >= 3.6

pytorch >= 1.4.0

transformers

numpy

## Data
Please unzip the data.zip file. It contains within it the Challenge Set. The Challenge Set’s examples are annotated with a 1 for high quality and a 0 or 2 for low quality. The code only reads in examples annotated with a 1 from the file.

## Code
Please run the commands contained in the "README Commands" file for data processing and reproduction of experiments.
