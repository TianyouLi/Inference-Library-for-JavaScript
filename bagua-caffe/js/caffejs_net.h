template <typename Dtype>
class JNet;

template <typename Dtype>
class NetOpener : public Nan::AsyncWorker {
public:
  NetOpener (Nan::Callback *cb, const char *file, const char *model) : Nan::AsyncWorker(cb), file_(file), model_(model) {}

  void Execute() {
    net_.reset(new Net<Dtype>(file_, TEST));
    net_->CopyTrainedLayersFrom(model_);
  }

  void HandleOKCallback() {
    Nan::HandleScope scope;
    JNet<Dtype> *jnet = new JNet<Dtype>(net_);

    MaybeLocal<Object> instance = Nan::NewInstance(Nan::New(JNet<Dtype>::ctor_instance_));
    jnet->Wrap(instance.ToLocalChecked());

    Local<Value> argv[] = { instance.ToLocalChecked() };
    callback->Call(1, argv);
  }

private:
  shared_ptr<Net<Dtype>> net_;
  std::string file_;
  std::string model_;
};

template <typename Dtype>
class NetForwarder : public Nan::AsyncWorker {
public:
  NetForwarder (Nan::Callback *cb, const shared_ptr<Net<Dtype>> &net) : Nan::AsyncWorker(cb), net_(net) {}

  void Execute() {
    net_->Forward();
  }

  void HandleOKCallback() {
    Nan::HandleScope scope;
    Local<Value> argv[] = { __js_vec(net_->output_blobs()) };
    callback->Call(1, argv);
  }

private:
  shared_ptr<Net<Dtype>> net_;
};

template <typename Dtype>
class JNet : public Nan::ObjectWrap {
public:
  explicit JNet() : net_(nullptr) {}

  explicit JNet(const shared_ptr<Net<Dtype>> &ref) : net_(ref) {
    init_persistents();
  }

  explicit JNet(const std::string &config, const std::string &model) : net_(new Net<Dtype>(config, TEST)) {
    net_->CopyTrainedLayersFrom(model);
    init_persistents();
  }

  static NAN_MODULE_INIT(Init) {
    Nan::HandleScope scope;
    Local<FunctionTemplate> ctor = Nan::New<FunctionTemplate>(New);
    ctor_.Reset(ctor);

    ctor->SetClassName(class_name());
    ctor->InstanceTemplate()->SetInternalFieldCount(1);
    Local<ObjectTemplate> proto = ctor->PrototypeTemplate();

    SetAccessor(proto, __js("blobs"), GetBlobs);

    SetMethod(proto, "forward", Forward);

    ctor_instance_.Reset(ctor->GetFunction());
    Nan::Set(target, class_name(), ctor->GetFunction());
  }

  Nan::Persistent<Object> blobs_p;

  template <typename D> friend class NetOpener;

private:
  void init_persistents() {
    auto names = net_->blob_names();
    auto blobs = net_->blobs();

    Nan::HandleScope scope;
    Local<Object> bo = Nan::New<Object>();
    for (size_t i = 0; i < names.size(); i++) {
      Nan::Set(bo, __js(names[i]), __js(blobs[i]));
    }
    blobs_p.Reset(bo);
  }

  static NAN_METHOD(New) {
    Nan::HandleScope scope;
    if (info.IsConstructCall() && info.Length() < 2) {
      return;
    }

    String::Utf8Value file(info[0]->ToString());
    String::Utf8Value model(info[1]->ToString());
    
    if (info.IsConstructCall()) {
      auto o = new JNet(*file, *model);
      o->Wrap(info.This());
      RETURN(info.This());
    }
    else if (info.Length() >= 3 && info[2]->IsFunction()) {
      auto cb = new Nan::Callback(info[2].As<Function>());
      Nan::AsyncQueueWorker(new NetOpener<Dtype>(cb, *file, *model));
    }
    else {
      Nan::ThrowError("Invalid argument");
    }
  }

  static NAN_GETTER(GetBlobs) {
    auto self = Unwrap<JNet>(info.This());
    RETURN(Nan::New(self->blobs_p));
  }

  static NAN_METHOD(Forward) {
    Nan::HandleScope scope;
    auto self = Unwrap<JNet>(info.This());
    Local<Value> fx = ARG(0);

    if (fx->IsFunction()) {
      auto cb = new Nan::Callback(fx.As<Function>());
      Nan::AsyncQueueWorker(new NetForwarder<Dtype>(cb, self->net_));
    }
    else {
      RETURN(__js_vec(self->net_->Forward()));
    }
  }

  #undef UNWRAP
  DECLARE_WRAP(Net, net)
