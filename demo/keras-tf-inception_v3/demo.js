"use strict";
var fs = require("fs");
var tf = require("../../bagua-tensorflow/src");

var graphDef = fs.readFileSync("../../model-zoo/tf/inception_v3/keras-tf-inception_v3.pb");
var image = tf.io.loadImage("../images/elephant.png");

var labels = require("../../model-zoo/tf/inception_v3/imagenet_class_index.json");

var transformer = new tf.io.Transformer([299, 299, 3]); // HWC

function preprocess(image) {
  var r = transformer.preprocess(image);
  let f = new Float32Array(r.data.buffer);
  for (let i = 0; i < f.length; i++) {
    f[i] = (f[i]/255 - 0.5) * 2;
  }
  return r;
}

//tf.io.namedWindow("Image");
//tf.io.show("Image", image);
//tf.io.waitKey();

function maxIndex(probabilities) {
  var best = 0;
  for (var i = 1; i < probabilities.length; ++i) {
    if (probabilities[i] > probabilities[best]) {
      best = i;
    }
  }
  return best;
}

var g = new tf.Graph(graphDef);
var sess = new tf.Session(g);

sess.run(["Softmax"],
  {
    input_1: new tf.Tensor(preprocess(image))
  },
  o => {
    let onehot = new Float32Array(o.data.buffer);
    let mi = maxIndex(onehot);
    console.log(labels[mi] ? labels[mi][1] : "---");
  }
);


