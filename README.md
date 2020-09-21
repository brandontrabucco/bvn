# Birkhoff-von Neumann Decomposition

This repository implements a fast batch-wise Birkhoff-von Neumann Decomposition with a Greedy Birkhoff Heuristic using TensorFlow 2.3. This implementation uses the greedy birkhoff heuristic in order to guarentee that the first k permutation matrices found by the algorithm have largest birkhoff coefficients. 

## Setup

You may install our package using this command.

```bash
pip install -e git+git://github.com/brandontrabucco/bvn.git#egg=bvn
```

## Usage

```python
import tensorflow as tf
import bvn

# create a doubly stochastic x
x = tf.fill([32, 5, 5], 0.2, name='doubly_stochastic')

# find its BvN decomposition
p, c = bvn.bvn(x, 100)

# permutations p and coefficients c approximate x
x == tf.reduce_sum(p * c[..., tf.newaxis, tf.newaxis], axis=1)
```

## Background

Refer to this paper for a mathematical justification: https://doi.org/10.1016/j.laa.2016.02.023
