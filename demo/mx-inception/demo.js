var fs = require("fs");
var mx = require('../../bagua-mxnet/src');

// load essentials

var image = mx.io.loadImage("../images/dog.jpg", 3);
var json_file = fs.readFileSync("../../model-zoo/mxnet/Inception-BN-symbol.json");
var param_file = fs.readFileSync("../../model-zoo/mxnet/Inception-BN-0126.params");
var nd_file = fs.readFileSync("../../model-zoo/mxnet/mean_224.nd");
var params = [224,224,3]

// preprocess

var transformer = new mx.io.Transformer(params);
transformer.setMeanValue(Array.prototype.slice.call(nd_file, 0)); // 0 ~ 255 => 0 ~ 1.0
var img = transformer.preprocess(image)


// create a pridictor

var prd = new mx.MPrd();

prd.create(params, json_file, param_file);
prd.setInput(img);
var output = prd.run();

// process output

var syn = loadSynset("../../model-zoo/mxnet/synset.txt")
console.log('Best Guess: ', syn[maxIndex(output)])

// utils

function maxIndex(probabilities) {
  var best = 0;
  for (var i = 1; i < probabilities.length; ++i) {
    if (probabilities[i] > probabilities[best]) {
      best = i;
    }
  }
  return best;
}

function loadSynset(path) {
  return fs.readFileSync(path).toString('ascii').split("\n").map(function (data) {
    return data.substring(10)
  })
}
