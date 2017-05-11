#include "mx.h"
#include "opencv.h"

// Mat
Nan::Persistent<FunctionTemplate> Mat::ctor_p;

static cv::Scalar ArrayToColor(Local<Object> a) {
  Local<Value> b = a->Get(0);
  Local<Value> g = a->Get(1);
  Local<Value> r = a->Get(2);

  return cv::Scalar(b->IntegerValue(), g->IntegerValue(), r->IntegerValue());
}

NAN_MODULE_INIT(Mat::Init)
{
  Nan::HandleScope scope;
  Local<FunctionTemplate> ctor = Nan::New<FunctionTemplate>(Mat::New);
  Mat::ctor_p.Reset(ctor);
  ctor->InstanceTemplate()->SetInternalFieldCount(1);
  ctor->SetClassName(__js("Mat"));

  Local<ObjectTemplate> proto = ctor->PrototypeTemplate();
  SetAccessor(proto, __js("cols"), GetCols);
  SetAccessor(proto, __js("rows"), GetRows);
  SetAccessor(proto, __js("channels"), GetChannels);
  SetAccessor(proto, __js("data"), GetData);

  SetMethod(proto, "rectangle", Rectangle);
}

NAN_METHOD(Mat::New) {
  Nan::HandleScope scope;
  auto mat = new Mat;

  mat->Wrap(info.This());
  RETURN(info.This());
}

NAN_GETTER(Mat::GetCols) {
  auto self = Unwrap<Mat>(info.This());
  RETURN(__js((int) self->mat_.cols));
}

NAN_GETTER(Mat::GetRows) {
  auto self = Unwrap<Mat>(info.This());
  RETURN(__js((int) self->mat_.rows));
}

NAN_GETTER(Mat::GetChannels) {
  auto self = Unwrap<Mat>(info.This());
  RETURN(__js((int) self->mat_.channels()));
}

static void jsab_dummy2 (char *data, void *holder) {}

NAN_GETTER(Mat::GetData) {
  auto self = Unwrap<Mat>(info.This());
  int sz = self->mat_.total()*self->mat_.elemSize();
  RETURN(Nan::NewBuffer((char *) self->mat_.data, sz, jsab_dummy2, nullptr).ToLocalChecked());
}

NAN_METHOD(Mat::Rectangle) {
  Nan::HandleScope scope;
  auto self = Unwrap<Mat>(info.This());
  int x0 = info[0]->IntegerValue();
  int y0 = info[1]->IntegerValue();
  int x1 = info[2]->IntegerValue();
  int y1 = info[3]->IntegerValue();
  cv::Scalar color(0, 0, 255);
  int thickness = 1;

  if (info.Length() > 4 && info[4]->IsArray()) {
    color = ArrayToColor(info[4]->ToObject());
  }

  if (info.Length() > 5 && info[5]->IntegerValue())
    thickness = info[5]->IntegerValue();

  cv::rectangle(self->mat_, cv::Point(x0, y0), cv::Point(x1, y1), color, thickness);
}

// Transformer
Transformer::Transformer (const std::vector<int> &shape) {
  height_ = shape[0];
  width_ = shape[1];
  num_channels_ = shape[2];
  scale_ = 0;
}

NAN_MODULE_INIT(Transformer::Init) {
  Nan::HandleScope scope;
  Local<FunctionTemplate> ctor = Nan::New<FunctionTemplate>(Transformer::New);
  ctor->InstanceTemplate()->SetInternalFieldCount(1);
  ctor->SetClassName(__js("Transformer"));

  Local<ObjectTemplate> proto = ctor->PrototypeTemplate();
  SetMethod(proto, "setMeanValue", SetMeanValue);
  SetMethod(proto, "setScale", SetScale);
  SetMethod(proto, "preprocess", Preprocess);

  Nan::Set(target, __js("Transformer"), ctor->GetFunction());
}

NAN_METHOD(Transformer::New) {
  Nan::HandleScope scope;
  auto t = new Transformer(__cxx_vec<int>(info[0]));

  t->Wrap(info.This());
  RETURN(info.This());
}

NAN_METHOD(Transformer::SetMeanValue) {
  auto self = Unwrap<Transformer>(info.This());
  auto values = __cxx_vec<int>(info[0]);

  std::vector<cv::Mat> ch;
  for (int i = 0; i < self->num_channels_; i++) {
    cv::Mat channel(self->height_, self->width_, CV_32FC1, cv::Scalar(values[i]));
    ch.push_back(channel);
  }
  cv::merge(ch, self->mean_);
  RETURN(info.This());
}

NAN_METHOD(Transformer::SetScale) {
  auto self = Unwrap<Transformer>(info.This());
  __cxx(info[0], self->scale_);

  RETURN(info.This());
}

NAN_METHOD(Transformer::Preprocess) {
  Nan::HandleScope scope;
  cv::Mat &img = Unwrap<Mat>(info[0]->ToObject())->mat_;
  auto self = Unwrap<Transformer>(info.This());

  printf("Transformer::Preprocess  %d => %d\n", img.channels(), self->num_channels_);
  cv::Mat src;
  if (img.channels() == 3 && self->num_channels_ == 1)
    cv::cvtColor(img, src, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && self->num_channels_ == 1)
    cv::cvtColor(img, src, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && self->num_channels_ == 3)
    cv::cvtColor(img, src, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && self->num_channels_ == 3)
    cv::cvtColor(img, src, cv::COLOR_GRAY2BGR);
  else
    src = img;

  cv::Mat resized;
  if (src.size().width != self->width_ || src.size().height != self->height_)
    cv::resize(src, resized, cv::Size(self->width_, self->height_));

  else
    resized = src;

  cv::Mat fmat;
  if (self->num_channels_ == 3)
    resized.convertTo(fmat, CV_32FC3);

  else
    resized.convertTo(fmat, CV_32FC1);


  cv::Mat normalized;
  if (self->mean_.empty()) 
    normalized = fmat;
  
  else 
    cv::subtract(fmat, self->mean_, normalized);


  if (self->scale_ != 0) {
    float *m = (float *) normalized.data;
    for (int i = 0; i < normalized.size().area()*self->num_channels_; i++) {
      m[i] *= self->scale_;
    }
  }


  cv::Mat res;
  normalized.convertTo(res, CV_8UC1);

  Local<Object> obj = Nan::New(Mat::ctor_p)->GetFunction()->NewInstance();
  Unwrap<Mat>(obj)->mat_ = res;
  RETURN (obj);
}

// VideoCapture
template <typename T>
class VCOpener : public Nan::AsyncWorker {
public:
  VCOpener (Nan::Callback *cb, const cv::VideoCapture &vc, T arg) : Nan::AsyncWorker(cb), vc_(vc), arg_(arg) {}

  void Execute() {
    res_ = vc_.open(arg_);
  }

  void HandleOKCallback() {
    Nan::HandleScope scope;
    Local<Value> argv[] = { __js(res_) };
    callback->Call(1, argv);
  }

private:
  cv::VideoCapture vc_;
  T arg_;
  bool res_;
};

class VCReader : public Nan::AsyncWorker {
public:
  VCReader (Nan::Callback *cb, const cv::VideoCapture &vc) : Nan::AsyncWorker(cb), vc_(vc) {}

  void Execute() {
    res_ = vc_.read(mat_);
  }

  void HandleOKCallback() {
    Nan::HandleScope scope;
    Local<Value> argv[1];

    if (res_) {
      Local<Object> obj = Nan::New(Mat::ctor_p)->GetFunction()->NewInstance();
      Nan::ObjectWrap::Unwrap<Mat>(obj)->mat_ = mat_;
      argv[0] = obj;
    }
    else {
      argv[0] = Nan::Undefined();
    }
    callback->Call(1, argv);
  }

private:
  cv::VideoCapture vc_;
  cv::Mat mat_;
  bool res_;
};


NAN_MODULE_INIT(VideoCapture::Init) {
  Nan::HandleScope scope;
  Local<FunctionTemplate> ctor = Nan::New<FunctionTemplate>(VideoCapture::New);
  ctor->InstanceTemplate()->SetInternalFieldCount(1);
  ctor->SetClassName(__js("VideoCapture"));

  Local<ObjectTemplate> proto = ctor->PrototypeTemplate();
  SetMethod(proto, "open", Open);
  SetMethod(proto, "read", Read);
  SetMethod(proto, "close", Close);

  Nan::Set(target, __js("VideoCapture"), ctor->GetFunction());
}

NAN_METHOD(VideoCapture::New) {
  Nan::HandleScope scope;
  auto t = new VideoCapture();

  t->Wrap(info.This());
  RETURN(info.This());
}

NAN_METHOD(VideoCapture::Open) {
  Nan::HandleScope scope;
  auto self = Unwrap<VideoCapture>(info.This());
  Local<Value> fx = ARG(1);

  if (fx->IsFunction()) {
    auto cb = new Nan::Callback(fx.As<Function>());
    Nan::AsyncWorker *worker;

    if (info[0]->IsNumber()) {
      worker = new VCOpener<int>(cb, self->vc_, info[0]->IntegerValue());
    }
    else {
      String::Utf8Value file(info[0]->ToString());
      worker = new VCOpener<std::string>(cb, self->vc_, *file);
    }
    Nan::AsyncQueueWorker(worker);
  }
  else {
    bool res;

    if (info[0]->IsString()) {
      String::Utf8Value file(info[0]->ToString());
      res = self->vc_.open(*file);
    }
    else {
      res = self->vc_.open(info[0]->IntegerValue());
    }
    RETURN(__js(res));
  }
}

NAN_METHOD(VideoCapture::Read) {
  Nan::HandleScope scope;
  auto self = Unwrap<VideoCapture>(info.This());
  Local<Value> fx = ARG(0);

  if (fx->IsFunction()) {
    auto cb = new Nan::Callback(fx.As<Function>());
    Nan::AsyncQueueWorker(new VCReader(cb, self->vc_));
  }
  else {
    cv::Mat mat;
    if (self->vc_.read(mat)) {
      Local<Object> obj = Nan::New(Mat::ctor_p)->GetFunction()->NewInstance();
      Nan::ObjectWrap::Unwrap<Mat>(obj)->mat_ = mat;
      RETURN(obj);
    }
  }
}

NAN_METHOD(VideoCapture::Close) {
  Nan::HandleScope scope;
  auto self = Unwrap<VideoCapture>(info.This());

  self->vc_.release();
}

// IO
static NAN_METHOD(LoadImage) {
  Nan::HandleScope scope;
  Local<Object> obj = Nan::New(Mat::ctor_p)->GetFunction()->NewInstance();
  Mat *img = Nan::ObjectWrap::Unwrap<Mat>(obj);
  int type = info.Length() >= 2 ? info[1]->IntegerValue() : int(CV_LOAD_IMAGE_UNCHANGED);

  try {
    if (info[0]->IsString()) {
      String::Utf8Value file(info[0]->ToString());
      img->mat_ = cv::imread(*file, type);
    }
    else if (node::Buffer::HasInstance(info[0])) {
      auto buf = (uint8_t *) node::Buffer::Data(info[0]);
      auto len = node::Buffer::Length(info[0]);

      cv::Mat t(len, 1, CV_32FC1, buf);
      img->mat_ = cv::imdecode(t, type);
    }
    RETURN (obj);
  } catch (cv::Exception& e) {
    Nan::ThrowError(Nan::Error(e.what()));
  }
}

static NAN_METHOD(NamedWindow) {
  Nan::HandleScope scope;
  String::Utf8Value name(info[0]->ToString());
  int flags = info.Length() > 1 ? info[1]->IntegerValue() : int(CV_WINDOW_AUTOSIZE);

  cv::namedWindow(*name, flags);
}

static NAN_METHOD(Show) {
  Nan::HandleScope scope;
  String::Utf8Value name(info[0]->ToString());
  cv::Mat &img = Nan::ObjectWrap::Unwrap<Mat>(info[1]->ToObject())->mat_;

  cv::imshow(*name, img);
}

static NAN_METHOD(WaitKey) {
  Nan::HandleScope scope;
  int k = cv::waitKey(info.Length() > 0 ? info[0]->IntegerValue() : 0);
  RETURN(__js(k));
}


NAN_MODULE_INIT(IO::Init) {
  Local<Object> io = Nan::New<Object>();

  Nan::Export(io, "loadImage", LoadImage);
  Nan::Export(io, "namedWindow", NamedWindow);
  Nan::Export(io, "show", Show);
  Nan::Export(io, "waitKey", WaitKey);

  Transformer::Init(io);
  VideoCapture::Init(io);

  Nan::Set(target, __js("io"), io);
}
