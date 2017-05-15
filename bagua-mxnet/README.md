
# How to build

## Step 1 - Prepare mxnet library

Download prebuilt mxnet library from our release.

put it under ./bagua-mxnet/lib/

## Step 2 - Modify enviroment variables

```shell
export INCLUDE_PATH=$INCLUDE_PATH:$PWD/include

export CPATH="$INCLUDE_PATH"

export CPPPATH="$INCLUDE_PATH"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/lib
```

## Step 3 - Build bagua-mxnet

```shell

cd bagua-mxnet

npm install

node-gyp rebuild


```

# Demo

Here we provided a Inception demo for image classification.

## Step 1 - Prepare training model

[download](https://drive.google.com/drive/folders/0B0o4NKGc64U5ZDNZdU04R0xqMWc?usp=sharing) training model.

put them under ./model-zoo/mxnet/

## Step 2 - Run the demo

```shell
cd demo/mx-inception

node demo.js
```