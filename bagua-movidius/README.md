
# How to build

## For Linux

Build the code. Please install libusb-1.0.

  ```shell
  sudo apt-get install libusb-1.0-0-dev
  ```

## For Windows
Please download the [Zadig](http://zadig.akeo.ie/downloads/zadig-2.3.exe) utility and manually install WinUSB driver for USB device VID_03E7&PID_F63B.
![Install WinUSB driver](https://github.com/TianyouLi/Inference-Library-for-JavaScript/raw/d7ab61c9eb6cf63f752bd8e3ac30aac1256a6d92/demo/NCSDK%20Zadig.png)

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

