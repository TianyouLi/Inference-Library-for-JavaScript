# How to build

## Step 1 - Prepare tensorflow

### Option 1 - How to build tensorflow from source code

*See [Download and Setup](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md) for instructions on how to install our release binaries or how to build from source.*

### Option 2 - Download prebuilt tensorflow

[Download](https://github.com/TianyouLi/bagua/releases/download/0.01/libtensorflow-lnx64.so) the prebuilt library.

put it under bagua-tensorflow/tensorflow.

## Step 2 - Modify enviroment variables

  ```shell
  cd bagua-tensorflow

  # Modify LD_LIBRARY_PATH

  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/tensorflow
  ```

## Step 3 - How to build bagua-tensorflow

  ```shell
  cd bagua-tensorflow

  npm install

  node-gyp configure build
  ```

# Demo

Here we provided a MNIST demo.


  ```shell
  cd demo/tf-mnist

  # Train and export mnist model, the model file will be stored under model-zoo/tf/mnist
  
  python mnist_train.py
  
  # Run demo

  node mnist.js
  ```