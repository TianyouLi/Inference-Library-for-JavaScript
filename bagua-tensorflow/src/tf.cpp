#include "tf.h"
#include "opencv.h"

static NAN_MODULE_INIT(Init) {
  Nan::HandleScope scope;

  // Tensorflow
  DataType::Init(target);
  Tensor::Init(target);
  Graph::Init(target);
  Session::Init(target);

  // Opencv
  Mat::Init(target);
  IO::Init(target);
}

NODE_MODULE(Tensorflow, Init);
