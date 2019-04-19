# Meta-Open-World-Learning
code for our TheWebConf (WWW 2019) paper titled "[Open-world Learning and Application to Product Classification](https://arxiv.org/abs/1809.06004)".

Checkou this [article](https://howardhsu.github.io/article/owl/) (or [this](https://howardhsu.github.io/article/overview/) ) for related background.

*** Feedbacks are welcomed to improve this repository ***

## Problem to Solve
Classic supervised learning assumes that the classes seen in testing must have appeared in training. However, this assumption is often violated in real-world applications when new topics emerge constantly or in e-commerce new categories of products appear daily.
A model working in such an open environment must be able to

(1) reject examples from unseen classes (not appeared during training) and 

(2) incrementally learn the new/unseen classes to expand the existing model. 

We call this problem open-world learning (OWL).

## Environment
This project is tested on Python 2.7 + Keras 2.2.2 with Tensorflow 1.4.0 on Ubuntu 16.04, but it should generally work for other versions.

## Data
Download the preprocessed data from [here](https://drive.google.com/file/d/1l0JR7u6FX4Av4Zf4mAFhQfBlbMFbq_1b/view?usp=sharing). Save it to amazon/data .
```
cd amazon/data/
tar -zxf data.tar.gz
```
This leads to the meta training classes in train1_npz and 3 sets (25, 50, 75 incrementally) of meta testing classes (including both training/testing data for meta testing classes).

We only release the preprocessed data at this stage (although some preprocessing code is available under amazon folder).


## Running Script
```
cd script
```
Single run with k=5 n=9 (ncls=9+1=10, with 1 positive class and 9 negative classes).
```
bash run.sh 5 10
```
Run all settings of k and n with 10 runs (note this will cause a lot of time so it's better to distribute tasks to different GPUs)
```
bash run_batch.sh
```
The results are saved in eval.json under each run's folder under different parameter setting.

## Citation:
```
@inproceedings{xu2019open,
  title={Open-world Learning and Application to Product Classification},
  author={Xu, Hu and Liu, Bing, Shu, Lei and Yu, P.},
  booktitle={Proceedings of the 2019 World Wide Web Conference on World Wide Web},
  year={2019},
  organization={International World Wide Web Conferences Steering Committee}
}
```

## TODO:
- [ ] preprocessing code and meta data.
