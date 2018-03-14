
# How to build

## For Linux

Build the code. Please install libusb-1.0.

  ```shell
  sudo apt-get install libusb-1.0-0-dev
  ```

## Build bagua-movidius

```shell
cd bagua-movidius

npm install

node-gyp configure build
```

# Demo

Here we provided a SSD demo for object detection on Movidius Neural Compute Stick.

## Run the demo


```shell
cd demo/movidius-ssd && node index.js
```

