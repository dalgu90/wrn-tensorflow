# wrn-tensorflow

A tensorflow implementation of Wide Residual Networks(https://arxiv.org/abs/1605.07146)

<b>Reference</b>:  

1. Original code: https://github.com/szagoruyko/wide-residual-networks  

<b>Prerequisite</b>

1. Python 3
1. TensorFlow 1.8+

<b>How To Run</b>

```shell
# Clone the repo.
git clone https://github.com/dalgu90/wrn-tensorflow.git
cd wrn-tensorflow

# Download CIFAR-100 dataset
wget https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
tar xvf cifar-100-binary.tar.gz

# Run! (WRN-28-1)
./train.sh

# To evaluate
./eval.sh wrn_28_1 4 1
```
