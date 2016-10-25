# wrn-tensorflow

A tensorflow implementation of Wide Residual Networks(https://arxiv.org/abs/1605.07146)

<b>Reference</b>:  

1. Original code: https://github.com/szagoruyko/wide-residual-networks  
2. The data module `cifar100_input.py` is from [TensorFlow tutorial](https://www.tensorflow.org/versions/master/tutorials/deep_cnn/index.html).

<b>Prerequisite</b>

1. TensorFlow

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
