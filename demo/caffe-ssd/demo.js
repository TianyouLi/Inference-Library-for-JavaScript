"use strict";

var caffe = require("../../bagua-caffe/build/Release/caffe.node");
var SSD = require("./ssd.js");

var LABELS = ["---", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"];

var image = caffe.io.loadImage("../images/dog.jpg");

caffe.io.namedWindow("Image");

function show (blob) {
  let detections = SSD.parse(blob, image);

  detections.forEach(o => {
    if (o.score > 0.3) {
      console.log(LABELS[o.label], o.score.toFixed(3), "@", o.xmin, o.ymin, o.xmax, o.ymax);
      image.rectangle(o.xmin, o.ymin, o.xmax, o.ymax);
    }
  });
  caffe.io.show("Image", image);
  caffe.io.waitKey();
}


var net = new caffe.Net('../../model-zoo/caffe/VOC0712/SSD_300x300/deploy.prototxt', '../../model-zoo/caffe/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel');

var transformer = new caffe.io.Transformer(net.blobs["data"].shape);
transformer.setMeanValue([104, 117, 123]);

net.blobs["data"].data = transformer.preprocess(image);

// Sync call
//show (net.forward()[0]);


// Async call
net.forward(output_blobs => {
  show (output_blobs[0]);
});

