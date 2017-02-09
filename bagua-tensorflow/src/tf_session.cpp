#include "tf.h"

IMPL_WRAP(Session);

Session::Session(TF_Session* s) {
  session_ = s;
}

Session::~Session() {
  if (session_ != nullptr) {
    TF_Status* status = TF_NewStatus();
    TF_CloseSession(session_, status);
    // Result of close is ignored, delete anyway.
    TF_DeleteSession(session_, status);
    TF_DeleteStatus(status);
  }
}

NAN_MODULE_INIT(Session::Init) {
  Nan::HandleScope scope;
  Local<FunctionTemplate> ctor = Nan::New<FunctionTemplate>(New);
  ctor_.Reset(ctor);

  ctor->SetClassName(class_name());
  ctor->InstanceTemplate()->SetInternalFieldCount(1);
  Local<ObjectTemplate> proto = ctor->PrototypeTemplate();

  SetMethod(proto, "run", run);
  SetMethod(proto, "close", close);

  ctor_instance_.Reset(ctor->GetFunction());
  Nan::Set(target, class_name(), ctor->GetFunction());
}

NAN_METHOD(Session::New) {
  Nan::HandleScope scope;
  TF_Graph* graph = Unwrap<Graph>(info[0]->ToObject())->graph_;
  TF_Status* status = TF_NewStatus();
  TF_SessionOptions* opts = TF_NewSessionOptions();
  TF_Session* session;

  if (info.Length() > 1 && info[1]->IsString()) {
    String::Utf8Value path(info[1]);
    const char *tags[] = {"serve"};
    session = TF_LoadSessionFromSavedModel(opts, nullptr, *path, tags, 1, graph, nullptr, status);
  }
  else {
    session = TF_NewSession(graph, opts, status);
  }

  if (TF_GetCode(status) != TF_OK) {
    DIE(TF_Message(status));
  }

  auto sess = new Session(session);
  sess->graph_ = graph;

  TF_DeleteSessionOptions(opts);
  TF_DeleteStatus(status);

  sess->Wrap(info.This());
  RETURN(info.This());
}

class SessRunnerArgs {
public:
  TF_Buffer *run_options = nullptr;
  std::vector<TF_Output>  inputs;
  std::vector<TF_Tensor*> input_values;
  std::vector<TF_Output>  outputs;
  std::vector<TF_Tensor*> output_values;
};

class SessRunner : public Nan::AsyncWorker {
public:
  SessRunner (Nan::Callback *cb, TF_Session *sess, std::shared_ptr<SessRunnerArgs> args) : Nan::AsyncWorker(cb), sess_(sess), args_(args) {}

  void Execute() {
    TF_Status *status = TF_NewStatus();
    TF_SessionRun (sess_,
        args_->run_options,
        &args_->inputs[0],
        &args_->input_values[0],
        args_->inputs.size(),
        &args_->outputs[0],
        &args_->output_values[0],
        args_->outputs.size(),
        nullptr,
        0,
        nullptr,
        status);

    success_ = TF_GetCode(status) == TF_OK;
    if (!success_) {
      printf("%s: %s", __func__, TF_Message(status));
    }
    TF_DeleteStatus(status);
  }

  void HandleOKCallback() {
    Nan::HandleScope scope;
    Local<Value> argv[1];

    if (!success_) {
      argv[0] = Nan::Undefined();
    }
    else {
      //printf("%s: %d bytes\n", __func__, (int)TF_TensorByteSize(args_->output_values[0]));
      argv[0] = Tensor::create(args_->output_values[0]);
    }

    callback->Call(1, argv);
  }

private:
  TF_Session *sess_;
  std::shared_ptr<SessRunnerArgs> args_;
  bool success_;
};

NAN_METHOD(Session::run) {
  Nan::HandleScope scope;
  auto sess = Unwrap<Session>(info.This());
  if (info.Length() < 2) {
    return;
  }

  std::shared_ptr<SessRunnerArgs> args (new SessRunnerArgs);

  Local<Array> fetches = Local<Array>::Cast(info[0]);
  for (size_t i = 0; i < fetches->Length(); i++) {
    Local<Value> v = fetches->Get(i);
    if (v->IsString()) {
      String::Utf8Value name(v);
      auto oper = TF_GraphOperationByName(sess->graph_, *name);
      if (!oper) {
        DIE("Non-exist operation");
        return;
      }
      args->outputs.push_back(TF_Output{oper, 0});
      args->output_values.push_back(nullptr);
    }
    else {
      DIE("UNIMPLEMENT");
    }
  }

  Local<FunctionTemplate> tensor_ctor = Nan::New(Tensor::ctor_);

  Local<Object> feed = info[1]->ToObject();
  Local<Array> feed_names = Nan::GetOwnPropertyNames(feed).ToLocalChecked();
  for (size_t i = 0; i < feed_names->Length(); i++) {
    Local<Value> key = feed_names->Get(i);

    String::Utf8Value name(key);
    auto oper = TF_GraphOperationByName(sess->graph_, *name);
    if (!oper) {
      DIE("Non-exist operation");
      return;
    }
    args->inputs.push_back(TF_Output{oper, 0});

    Local<Value> val = feed->Get(key);
    if (!tensor_ctor->HasInstance(val)) {
      DIE("Feeding invalid tensor");
      return;
    }

    auto tensor = Unwrap<Tensor>(val->ToObject());
    args->input_values.push_back(tensor->tensor_);
  }

  // Async call
  if (info.Length() > 2 && info[info.Length() - 1]->IsFunction()) {
    auto cb = new Nan::Callback(info[info.Length() - 1].As<Function>());
    Nan::AsyncQueueWorker(new SessRunner(cb, sess->session_, args));
    return;
  }

  // Sync call
  TF_Status* status = TF_NewStatus();
  TF_SessionRun (sess->session_,
    args->run_options,
    &args->inputs[0],
    &args->input_values[0],
    args->inputs.size(),
    &args->outputs[0],
    &args->output_values[0],
    args->outputs.size(),
    nullptr,
    0,
    nullptr,
    status);

  if (TF_GetCode(status) != TF_OK) {
    printf("%s: %s", __func__, TF_Message(status));
  }
  else {
    //printf("%s: %d bytes\n", __func__, (int)TF_TensorByteSize(args->output_values[0]));
    RETURN(Tensor::create(args->output_values[0]));
  }
  TF_DeleteStatus(status);
}

NAN_METHOD(Session::close) {
  DIE("UNIMPLEMENT");
}
