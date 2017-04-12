#include "caffejs.h"
#include "caffejs_mat.h"
#include "caffejs_blob.h"
#include "caffejs_net.h"
#include "caffejs_io.h"


static NAN_METHOD(getMode) {
  Nan::HandleScope scope;
  RETURN(__js(Caffe::mode() == Caffe::CPU ? "CPU" : "GPU"));
}

static NAN_METHOD(setMode) {
  Nan::HandleScope scope;
  String::Utf8Value mode(info[0]->ToString());

  if (strcasecmp(*mode, "CPU") == 0) {
    Caffe::set_mode(Caffe::CPU);
  }
  else if (strcasecmp(*mode, "GPU") == 0) {
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(info.Length() > 1 ? info[1]->IntegerValue() : 0);
  }
  else {
    Nan::ThrowError("Invalid argument");
  }
}


static NAN_MODULE_INIT(Init) {
  JBlob<float>::Init(target);
  JNet<float>::Init(target);

  JMat::Init(target);
  JIO::Init(target);

  Nan::Export(target, "getMode", getMode);
  Nan::Export(target, "setMode", setMode);
}

NODE_MODULE(Caffe, Init);
