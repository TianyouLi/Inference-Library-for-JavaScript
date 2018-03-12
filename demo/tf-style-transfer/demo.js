"use strict";
var fs = require("fs");
var tf = require("../../bagua-tensorflow/src");

var MODEL_NAMES = ["la_muse", "rain_princess", "scream", "udnie", "wave", "wreck"];

var MODELS = MODEL_NAMES.map(model => {
  let graphDef = fs.readFileSync("/home/kanghua/git/bagua/demo/tf-style-transfer/models/" + model + ".pb");
  let sess = new tf.Session(new tf.Graph(graphDef));
  return sess;
});

var STYLES = MODEL_NAMES.map(model => tf.io.loadImage("/home/kanghua/git/bagua/demo/tf-style-transfer/styles/" + model + ".jpg"));

var IMAGES = ["cat.png", "chicago.jpg", "elephant.png", "fox.png", "stata.jpg"];

var MODEL_IDX = 0;
var IMAGE_IDX = 0;

tf.io.namedWindow("Style", 2);
tf.io.resizeWindow("Style", 320, 320);

tf.io.namedWindow("Source", 2);
tf.io.resizeWindow("Source", 320, 320);

function Loop() {
  var image = tf.io.loadImage("/home/kanghua/git/bagua/demo/tf-style-transfer/images/" + IMAGES[IMAGE_IDX]);

  tf.io.show("Style", STYLES[MODEL_IDX]);
  tf.io.show("Source", image);
  image.cvtColor(4);

  var W = image.cols, H = image.rows;
  var transformer = new tf.io.Transformer([H, W, 3]);

  var o = MODELS[MODEL_IDX].run(["add_37"],
    {
      "input": new tf.Tensor(transformer.preprocess(image)),
      "stack": new tf.Tensor([1, (H/2 + 0.5) | 0, (W/2 + 0.5) | 0, 64]),
      "stack_1": new tf.Tensor([1, H, W, 32])
    });

  let om = tf.io.newImage(H, W, new Float32Array(o.data.buffer));
  om.cvtColor(4).resize(image.rows, image.cols);
  tf.io.namedWindow("Image");
  tf.io.show("Image", om);

  let key = tf.io.waitKey(process.argv[2]);
  if (key < 0) {
    IMAGE_IDX++;
    if (IMAGE_IDX === IMAGES.length) {
      IMAGE_IDX = 0;
      MODEL_IDX++;
      if (MODEL_IDX == MODELS.length) {
        MODEL_IDX = 0;
      }
    }
  }
  else {
    //console.log(key & 0xff);
    switch(key & 0xff) {
    case 27:
    case 'q':
    case 'Q':
      process.exit(0);
      return;
    case 83: // right
      IMAGE_IDX++;
      if (IMAGE_IDX >= IMAGES.length)
        IMAGE_IDX = 0;
      break;
    case 81: // left
      if (IMAGE_IDX > 0)
        IMAGE_IDX--;
      else
        IMAGE_IDX = IMAGES.length - 1;
      break;
    case 84: // down
      MODEL_IDX++;
      if (MODEL_IDX >= MODELS.length)
        MODEL_IDX = 0;
      break;
    case 82: // up
      if (MODEL_IDX > 0)
        MODEL_IDX--;
      else
        MODEL_IDX = MODELS.length - 1;
      break;
    }
  }
  process.nextTick(Loop);
}

Loop();
