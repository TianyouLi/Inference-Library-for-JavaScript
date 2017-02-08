#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Mat: public Nan::ObjectWrap {
public:
  cv::Mat mat_;
  static Nan::Persistent<FunctionTemplate> ctor_p;

  static NAN_MODULE_INIT(Init);
  static NAN_METHOD(New);
  static NAN_GETTER(GetCols);
  static NAN_GETTER(GetRows);
  static NAN_GETTER(GetChannels);
  static NAN_GETTER(GetData);
  static NAN_METHOD(Rectangle);
};

class Transformer: public Nan::ObjectWrap {
public:
  cv::Mat mean_;
  int num_channels_;
  int height_;
  int width_;
  float scale_;

  Transformer (const std::vector<int> &shape);
  static NAN_MODULE_INIT(Init);
  static NAN_METHOD(New);
  static NAN_METHOD(SetMeanValue);
  static NAN_METHOD(SetScale);
  static NAN_METHOD(Preprocess);
};

class VideoCapture: public Nan::ObjectWrap {
public:
  cv::VideoCapture vc_;

  static NAN_MODULE_INIT(Init);
  static NAN_METHOD(New);
  static NAN_METHOD(Open);
  static NAN_METHOD(Read);
  static NAN_METHOD(Close);
};

class IO {
public:
  static NAN_MODULE_INIT(Init);
};
