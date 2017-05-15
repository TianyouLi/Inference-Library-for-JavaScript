
# How to build

## Step 1 - Prepare caffe library

Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.

  ```shell
  # Modify Makefile.config according to your Caffe installation.

  make lib -j8
  ```

## Step 2 - Modify enviroment variables

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/__build/lib:/usr/local/cuda/lib64
```

## Step 3 - Build bagua-caffe

```shell
cd bagua-mxnet

npm install

node-gyp configure build
```

# Demo

Here we provided a SSD demo for object detection.

## Step 1 - Prepare caffemodel


Download [SSD300*](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712_SSD_300x300.tar.gz), and only extract the **caffemodel** file into model-zoo/caffe/VOC0712/SSD_300x300.


## Step 2 - Run the demo


```shell
cd demo/caffe-ssd && node demo.js
```

# Q&A

## 1. Error: Module version mismatch. Expected 48, got 46.

```shell
rm -r ./bagua-caffe/node_modules
npm update
npm cache clean
npm insall
```

if the error still shows up, change node version to 4+.