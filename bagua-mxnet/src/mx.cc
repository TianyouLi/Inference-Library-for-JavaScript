#include "mx.h"
#include "opencv.h"


using v8::FunctionTemplate;



// NativeExtension.cc represents the top level of the module.
// C++ constructs that are exposed to javascript are exported here

NAN_MODULE_INIT(InitAll) {
  // Nan::Set(target, Nan::New("createPred").ToLocalChecked(),
  //   Nan::GetFunction(Nan::New<FunctionTemplate>(createPred)).ToLocalChecked());
  Mat::Init(target);
  IO::Init(target);
  MPrd::Init(target);

}

NODE_MODULE(mxnet, InitAll)
