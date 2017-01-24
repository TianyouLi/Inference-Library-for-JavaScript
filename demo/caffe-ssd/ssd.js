exports.parse = function (blob, image) {
  var w = image.cols, h = image.rows;
  
  var f = new Float32Array(blob.data.buffer);
  var num_det = blob.shape[2];
  var out = [];

  for (var i = 0; i < num_det; i++) {
    if (f[i * 7] >= 0) {
      var o = {
        label: f[i * 7 + 1] | 0,
        score: f[i * 7 + 2],
        xmin: (f[i * 7 + 3] * w) | 0,
        ymin: (f[i * 7 + 4] * h) | 0,
        xmax: (f[i * 7 + 5] * w) | 0,
        ymax: (f[i * 7 + 6] * h) | 0
      };
      out.push(o);
    }
  }

  return out;
};
