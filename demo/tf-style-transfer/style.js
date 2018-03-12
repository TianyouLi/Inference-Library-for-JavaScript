"use strict";
var fs = require("fs");
var tf = require("../../bagua-tensorflow/src");

var MODELS = ["la_muse", "rain_princess", "scream", "udnie", "wave", "wreck"];
var IMAGES = ["cat.png", "chicago.jpg", "elephant.png", "fox.png", "stata.jpg"];

var MODEL_IDX = 0;
var IMAGE_IDX = 0;

var graphDef = fs.readFileSync("/home/kanghua/git/bagua/demo/tf-style-transfer/models/" + (process.argv[2] || "udnie") + ".pb");
var image = tf.io.loadImage("/home/kanghua/git/bagua/demo/tf-style-transfer/images/" + (process.argv[3] || "cat.png"));
var style = tf.io.loadImage("/home/kanghua/git/bagua/demo/tf-style-transfer/styles/" + (process.argv[2] || "udnie") + ".jpg");

tf.io.namedWindow("Style", 2);
tf.io.resizeWindow("Style", 320, 320);
tf.io.show("Style", style);

tf.io.namedWindow("Source", 2);
tf.io.resizeWindow("Source", 320, 320);
tf.io.show("Source", image);
image.cvtColor(4);

var W = image.cols, H = image.rows;
var transformer = new tf.io.Transformer([H, W, 3]);
var sess = new tf.Session(new tf.Graph(graphDef));

sess.run(["add_37"],
  {
    "input": new tf.Tensor(transformer.preprocess(image)),
    "stack": new tf.Tensor([1, (H/2 + 0.5) | 0, (W/2 + 0.5) | 0, 64]),
    "stack_1": new tf.Tensor([1, H, W, 32])
  },
  o => {
    let om = tf.io.newImage(H, W, new Float32Array(o.data.buffer));
    om.cvtColor(4).resize(image.rows, image.cols);
    tf.io.namedWindow("Image");
    tf.io.show("Image", om);
    let key = tf.io.waitKey(process.argv[4]);
    console.log(key);
  }
);

