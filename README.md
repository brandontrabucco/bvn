# Birkhoff-von Neumann Decomposition


This repository implements a fast batch-wise Birkhoff-von Neumann Decomposition with a Greedy Birkhoff Heuristic using TensorFlow 2.3. This implementation uses the greedy birkhoff heuristic in order to guarentee that the first k permutation matrices found by the algorithm have largest birkhoff coefficients. 

## Usage

```python
import bvn

# create a doubly stochastic x
x = tf.ones([32, 5, 5])
x = x / tf.reduce_sum(x, axis=1, keepdims=True)

# find its BvN decomposition
p, c = bvn.birkhoff_von_neumann(x)

# permutations p and coefficients c approximate x
x == tf.reduce_sum(p * c[..., tf.newaxis, tf.newaxis], axis=1)
```

## Dependencies

This package is meant for use in TensorFlow and requires the following packages to be installed in python.

```bash
pip install tensorflow==2.3
pip install networkx
```

## Background

Refer to this paper for a mathematical justification: https://doi.org/10.1016/j.laa.2016.02.023
