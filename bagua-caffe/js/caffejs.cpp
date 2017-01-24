#include "caffejs.h"
#include "caffejs_mat.h"
#include "caffejs_blob.h"
#include "caffejs_net.h"
#include "caffejs_io.h"

static NAN_MODULE_INIT(Init) {
  JBlob<float>::Init(target);
  JNet<float>::Init(target);

  JMat::Init(target);
  JIO::Init(target);
}

NODE_MODULE(Caffe, Init);
