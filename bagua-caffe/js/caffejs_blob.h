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
    auto shape = __cxx_vec<int>(value);
    blob_->Reshape(shape);
  }

  static NAN_GETTER(GetData) {
    UNWRAP;
    RETURN(__js_buffer(blob_->data(), blob_->count() * sizeof(Dtype)));
  }


  static Dtype * store (Dtype *data, cv::Mat &mat, int h, int w, int c) {
    std::vector<cv::Mat> channels;

    // verify mat.rows == h and mat.cols == w
    for (int i = 0; i < c; i++) {
      cv::Mat channel(h, w, sizeof(Dtype) == sizeof(float) ? CV_32FC1 : CV_64FC1, data);
      channels.push_back(channel);
      data += w * h;
    }

    cv::split(mat, channels);
    return data;
  }

  static Dtype * store (Dtype *data, Local<Value> value, int h, int w, int c) {
    Nan::HandleScope scope;
    Local<FunctionTemplate> mat_ctor = Nan::New(mat_ctor_p);
    
    if (value->IsArrayBuffer()) {
      Local<ArrayBuffer> ab = Local<ArrayBuffer>::Cast(value);
      auto contents = ab->GetContents();
      
      memcpy (data, contents.Data(), contents.ByteLength());
      data += c * w * h;
    }
    else if (value->IsTypedArray()) {
      Local<TypedArray> a = Local<TypedArray>::Cast(value);
      Local<ArrayBuffer> ab = a->Buffer();
      auto contents = ab->GetContents();
      
      memcpy (data, contents.Data(), contents.ByteLength());
      data += c * w * h;
    }
    else if (mat_ctor->HasInstance(value)) {
      cv::Mat &mat = Unwrap<JMat>(value->ToObject())->mat_;
      
      data = store (data, mat, h, w, c);
    }
    else {
      Nan::ThrowError("Invalid argument");
    }
    return data;
  }

  static NAN_SETTER(SetData) {
    Nan::HandleScope scope;
    UNWRAP;

    Dtype *data = blob_->mutable_cpu_data();
    int h = blob_->height();
    int w = blob_->width();
    int c = blob_->channels();

    auto shape = blob_->shape();
    //printf("Shape: %d, %d, %d, %d\n", shape[0], shape[1], shape[2], shape[3]);
    
    if (value->IsArray() && !value->IsTypedArray()) {
      Local<Array> a = Local<Array>::Cast(value);
      //printf(" Batch: %d\n", a->Length());
      if (shape[0] != (int) a->Length()) {
        Nan::ThrowError("Batch size unmatched");
        return;
      }

      for (int i = 0; i < (int) a->Length(); i++) {
        data = store (data, a->Get(i), h, w, c);
      }
    }
    else {
      if (shape[0] != 1) {
        Nan::ThrowError("Batch size unmatched");
        return;
      }

      store (data, value, h, w, c);
    }
  }

  #undef UNWRAP
  DECLARE_WRAP(Blob, blob)

