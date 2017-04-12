class JTransformer: public Nan::ObjectWrap {
public:
  cv::Mat mean_;
  int num_channels_;
  int height_;
  int width_;
  float scale_;

  JTransformer (const std::vector<int> &shape) {
    num_channels_ = shape[1];
    height_ = shape[2];
    width_ = shape[3];
    scale_ = 0;
  }

  static NAN_MODULE_INIT(Init) {
    Nan::HandleScope scope;
    Local<FunctionTemplate> ctor = Nan::New<FunctionTemplate>(JTransformer::New);
    ctor->InstanceTemplate()->SetInternalFieldCount(1);
    ctor->SetClassName(__js("Transformer"));

    Local<ObjectTemplate> proto = ctor->PrototypeTemplate();
    SetMethod(proto, "setMeanValue", SetMeanValue);
    SetMethod(proto, "setScale", SetScale);
    SetMethod(proto, "preprocess", Preprocess);

    Nan::Set(target, __js("Transformer"), ctor->GetFunction());
  }

  static NAN_METHOD(New) {
    Nan::HandleScope scope;
    auto t = new JTransformer(__cxx_vec<int>(info[0]));

    t->Wrap(info.This());
    RETURN(info.This());
  }

  static NAN_METHOD(SetMeanValue) {
    auto self = Unwrap<JTransformer>(info.This());
    auto values = __cxx_vec<int>(info[0]);

    std::vector<cv::Mat> ch;
    for (int i = 0; i < self->num_channels_; i++) {
      cv::Mat channel(self->height_, self->width_, CV_32FC1, cv::Scalar(values[i]));
      ch.push_back(channel);
    }
    cv::merge(ch, self->mean_);
    RETURN(info.This());
  }

  static NAN_METHOD(SetScale) {
    auto self = Unwrap<JTransformer>(info.This());
    __cxx(info[0], self->scale_);

    RETURN(info.This());
  }

  static NAN_METHOD(Preprocess) {
    Nan::HandleScope scope;
    cv::Mat &img = Unwrap<JMat>(info[0]->ToObject())->mat_;
    auto self = Unwrap<JTransformer>(info.This());

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
    if (self->mean_.empty()) {
      normalized = fmat;
    }
    else {
      cv::subtract(fmat, self->mean_, normalized);
    }

    if (self->scale_ != 0) {
      float *m = (float *) normalized.data;
      for (int i = 0; i < normalized.size().area(); i++) {
        m[i] *= self->scale_;
      }
    }

    MaybeLocal<Object> obj = Nan::NewInstance(Nan::New(mat_ctor_instance_p));
    Unwrap<JMat>(obj.ToLocalChecked())->mat_ = normalized;
    RETURN (obj.ToLocalChecked());
  }
};

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
      MaybeLocal<Object> obj = Nan::NewInstance(Nan::New(mat_ctor_instance_p));
      Nan::ObjectWrap::Unwrap<JMat>(obj.ToLocalChecked())->mat_ = mat_;
      argv[0] = obj.ToLocalChecked();
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

class JVideoCapture: public Nan::ObjectWrap {
public:
  cv::VideoCapture vc_;

  static NAN_MODULE_INIT(Init) {
    Nan::HandleScope scope;
    Local<FunctionTemplate> ctor = Nan::New<FunctionTemplate>(JVideoCapture::New);
    ctor->InstanceTemplate()->SetInternalFieldCount(1);
    ctor->SetClassName(__js("VideoCapture"));

    Local<ObjectTemplate> proto = ctor->PrototypeTemplate();
    SetMethod(proto, "open", Open);
    SetMethod(proto, "read", Read);
    SetMethod(proto, "close", Close);

    Nan::Set(target, __js("VideoCapture"), ctor->GetFunction());
  }

  static NAN_METHOD(New) {
    Nan::HandleScope scope;
    auto t = new JVideoCapture();

    t->Wrap(info.This());
    RETURN(info.This());
  }

  static NAN_METHOD(Open) {
    Nan::HandleScope scope;
    auto self = Unwrap<JVideoCapture>(info.This());
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

  static NAN_METHOD(Read) {
    Nan::HandleScope scope;
    auto self = Unwrap<JVideoCapture>(info.This());
    Local<Value> fx = ARG(0);

    if (fx->IsFunction()) {
      auto cb = new Nan::Callback(fx.As<Function>());
      Nan::AsyncQueueWorker(new VCReader(cb, self->vc_));
    }
    else {
      cv::Mat mat;
      if (self->vc_.read(mat)) {
        MaybeLocal<Object> obj = Nan::NewInstance(Nan::New(mat_ctor_instance_p));
        Nan::ObjectWrap::Unwrap<JMat>(obj.ToLocalChecked())->mat_ = mat;
        RETURN(obj.ToLocalChecked());
      }
    }
  }

  static NAN_METHOD(Close) {
    Nan::HandleScope scope;
    auto self = Unwrap<JVideoCapture>(info.This());

    self->vc_.release();
  }
};

static NAN_METHOD(LoadImage) {
  Nan::HandleScope scope;
  MaybeLocal<Object> obj = Nan::NewInstance(Nan::New(mat_ctor_instance_p));
  JMat *img = Nan::ObjectWrap::Unwrap<JMat>(obj.ToLocalChecked());
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
    RETURN (obj.ToLocalChecked());
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
  cv::Mat &img = Nan::ObjectWrap::Unwrap<JMat>(info[1]->ToObject())->mat_;

  cv::imshow(*name, img);
}

static NAN_METHOD(WaitKey) {
  Nan::HandleScope scope;
  int k = cv::waitKey(info.Length() > 0 ? info[0]->IntegerValue() : 0);
  RETURN(__js(k));
}


class JIO {
public:
  static NAN_MODULE_INIT(Init) {
    Local<Object> io = Nan::New<Object>();
    
    Nan::Export(io, "loadImage", LoadImage);
    Nan::Export(io, "namedWindow", NamedWindow);
    Nan::Export(io, "show", Show);
    Nan::Export(io, "waitKey", WaitKey);
    
    JTransformer::Init(io);
    JVideoCapture::Init(io);

    Nan::Set(target, __js("io"), io);
  }
};
