#include "tf.h"

NAN_MODULE_INIT(DataType::Init) {
  Nan::HandleScope scope;
  Nan::Set(target, __js("float32"), __js(TF_FLOAT));
  Nan::Set(target, __js("float64"), __js(TF_DOUBLE));
  Nan::Set(target, __js("int32"), __js(TF_INT32));
  Nan::Set(target, __js("int8"), __js(TF_INT8));
  Nan::Set(target, __js("uint8"), __js(TF_UINT8));
  Nan::Set(target, __js("int16"), __js(TF_INT16));
  Nan::Set(target, __js("uint16"), __js(TF_UINT16));
  Nan::Set(target, __js("bool"), __js(TF_BOOL));
}
