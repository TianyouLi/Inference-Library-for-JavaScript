struct SMHolder {
  SMHolder(const shared_ptr<SyncedMemory> &a) : sm(a) {}
  shared_ptr<SyncedMemory> sm;
};

static void jsab_cb (char *data, void *holder) {
  delete reinterpret_cast<SMHolder *>(holder);
}

static Local<Object> __js_buffer (const shared_ptr<SyncedMemory> &sm, size_t size) {
  return Nan::NewBuffer((char *) sm->cpu_data(), size, jsab_cb, new SMHolder(sm)).ToLocalChecked();
}

template <typename Dtype>
static Local<Value> __js (Blob<Dtype> *blob) {
  Nan::EscapableHandleScope scope;
  Local<Object> o = Nan::New<Object>();

  Nan::Set(o, __js("shape"), __js_vec(blob->shape()));
  Nan::Set(o, __js("data"), __js_buffer(blob->data(), blob->count() * sizeof(Dtype)));
  return scope.Escape(o);
}


template <typename Dtype>
class JBlob : public Nan::ObjectWrap {
public:
  explicit JBlob(const shared_ptr<Blob<Dtype>> &ref) : blob_(ref) {}

  explicit JBlob(const std::vector<int> &shape) : blob_(new Blob<Dtype>(shape)) {
    Nan::AdjustExternalMemory(blob_->count() * 2 * sizeof(Dtype));
  }

  virtual ~JBlob() {
    if (blob_.use_count() == 1)
      Nan::AdjustExternalMemory(-blob_->count() * 2 * sizeof(Dtype));
  }

  static NAN_MODULE_INIT(Init) {
    Nan::HandleScope scope;
    Local<FunctionTemplate> ctor = Nan::New<FunctionTemplate>(New);
    ctor_.Reset(ctor);

    ctor->SetClassName(class_name());
    ctor->InstanceTemplate()->SetInternalFieldCount(1);
    Local<ObjectTemplate> proto = ctor->PrototypeTemplate();

    SetAccessor(proto, __js("shape"), GetShape, SetShape);
    SetAccessor(proto, __js("data"), GetData, SetData);

    ctor_instance_.Reset(ctor->GetFunction());
    Nan::Set(target, class_name(), ctor->GetFunction());
  }

private:
  static NAN_METHOD(New) {
    Nan::HandleScope scope;

    if (info.IsConstructCall()) {
      auto shape = __cxx_vec<int>(ARG(0));
      JBlob *o = new JBlob(shape);
      o->Wrap(info.This());
      RETURN(info.This());
    }
  }

  #define UNWRAP \
    auto blob_ = Unwrap<JBlob>(info.This())->blob_

  static NAN_GETTER(GetShape) {
    UNWRAP;
    RETURN(__js_vec(blob_->shape()));
  }

  static NAN_SETTER(SetShape) {
    UNWRAP;
    blob_->Reshape(__cxx_vec<int>(value));
  }

  static NAN_GETTER(GetData) {
    UNWRAP;
    RETURN(__js_buffer(blob_->data(), blob_->count() * sizeof(Dtype)));
  }

  static NAN_SETTER(SetData) {
    UNWRAP;
    cv::Mat &img = Unwrap<JMat>(value->ToObject())->mat_;

    std::vector<cv::Mat> channels;
    Dtype *data = blob_->mutable_cpu_data();

    for (int i = 0; i < blob_->channels(); i++) {
      cv::Mat channel(blob_->height(), blob_->width(), sizeof(Dtype) == sizeof(float) ? CV_32FC1 : CV_64FC1, data);
      channels.push_back(channel);
      data += blob_->width() * blob_->height();
    }

    cv::split(img, channels);
  }

  #undef UNWRAP
  DECLARE_WRAP(Blob, blob)

