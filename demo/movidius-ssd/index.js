const fs = require("fs");
const cv = require("opencv");
const mvnc = require("../../bagua-movidius/build/Release/ncsdk.node");

var win = new cv.NamedWindow('Movidius', 0);

const LABELS = [ "background",
  "aeroplane", "bicycle", "bird", "boat",
  "bottle", "bus", "car", "cat", "chair",
  "cow", "diningtable", "dog", "horse",
  "motorbike", "person", "pottedplant",
  "sheep", "sofa", "train", "tvmonitor"];

const networkDim = 300;

function image_to_tensor(image) {
  var copy = image.clone();
  var size = image.size();

  copy.resize(networkDim, networkDim);

  var data = copy.getData();

  var fp32 = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    fp32[i] = (data[i] - 127.5) * 0.007843;
  }
  return fp32;
}

function detect(graph, image) {

  graph.LoadTensor(image_to_tensor(image));

  var size = image.size();
  var output = graph.GetResult();

  var num_valid_boxes = output[0];
  for (let i = 0; i < num_valid_boxes; i++) {
    let idx = 7 + i * 7;
    if (!isFinite(output[idx]) ||
        !isFinite(output[idx + 1]) ||
        !isFinite(output[idx + 3]) ||
        !isFinite(output[idx + 3]) ||
        !isFinite(output[idx + 4]) ||
        !isFinite(output[idx + 5]) ||
        !isFinite(output[idx + 6]))
        continue;

    let x1 = output[idx + 3] * size[1] | 0;
    let y1 = output[idx + 4] * size[0] | 0;
    let x2 = output[idx + 5] * size[1] | 0;
    let y2 = output[idx + 6] * size[0] | 0;
    if (x1 < 0 || y1 < 0 || x2 < 0 || y2 < 0 ||
        x1 > size[1] || y1 > size[0] || x2 > size[1] || y2 > size[0])
        continue;

    let confidence = output[idx + 2] * 100;
    if (confidence < 30.0)
      continue;

    let clazz = output[idx + 1] | 0;
    console.log(LABELS[clazz], confidence, x1, y1, x2, y2);
    image.rectangle([x1, y1], [x2 - x1, y2 - y1]);
    image.putText(LABELS[clazz], x1, y1 + 10, "HERSEY_SIMPLEX", 0, 0.5, 0.5);
  }
}

try {
  var devices = mvnc.EnumerateDevices();
  if (!devices || devices.length === 0) {
    console.log("No found Movidius Neural Compute Stick.");
    return process.exit(0);
  }

  console.log("Found " + devices.length + " Movidius Neural Compute Stick.");
  var device = devices[0];
  if (!device.OpenDevice()) {
    console.log("Failed to open Neural Compute Stick.");
    return process.exit(0);
  }

  var graphBuffer = fs.readFileSync("mobilenet.graph");
  //var graphBuffer = fs.readFileSync("SSD_300x300.graph");
  var graph = device.AllocateGraph(graphBuffer);
  if (!graph) {
    device.CloseDevice();
    console.log("Failed to call AllocateGraph.");
    return process.exit(0);
  }

  /*cv.readImage("./fish-bike.jpg", (err, image) => {
    if (err) {
      graph.DeallocateGraph();
      device.CloseDevice();
      console.log("Failed to load image.");
      return process.exit(0);
    }

    detect(graph, image);

    graph.DeallocateGraph();
    device.CloseDevice();

    win.show(image);
    win.blockingWaitKey(0);
  });*/

  var camera = new cv.VideoCapture(0);

  function fetch() {
    camera.read(function(err, im) {
      if (err) {
        graph.DeallocateGraph();
        device.CloseDevice();
        console.log("Failed to read camera.");
        return process.exit(0);
      }

      var size = im.size();
      if (size[0] > 0 && size[1] > 0){
        detect(graph, im);
        win.show(im);
      }

      var key = win.blockingWaitKey(5) & 0xff;
      if (key == 'Q' || key == 'q' || key == 27) {
        camera.release();
        graph.DeallocateGraph();
        device.CloseDevice();
        process.exit(0);
        return;
      }

      fetch();
    });
  }

  fetch();

} catch (e){
  console.log(e);
}
