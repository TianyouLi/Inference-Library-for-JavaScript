#include <node.h>
#include <nan.h>
#include <iostream>
#include <string.h>
#include <sstream>
#include <vector>
#include <list>
#include <memory>
#include <fstream>
#include "mxnet/c_api.h"
#include "mxnet/c_predict_api.h"
#include "macro.h"

class MPrd : public Nan::ObjectWrap {
public:
  MPrd();

  static NAN_MODULE_INIT(Init);
  static NAN_METHOD(New);
  static NAN_METHOD(create);
  static NAN_METHOD(run);
  static NAN_METHOD(setInput);
  PredictorHandle pred_hnd;
  NDListHandle nd_hnd;
  const mx_float *nd_data;
  DECLARE_WRAP(MPrd);
};
