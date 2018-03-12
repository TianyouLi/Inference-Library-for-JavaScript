#include "tf.h"

IMPL_WRAP(Graph);

Graph::Graph() {
  graph_ = TF_NewGraph();
}

Graph::~Graph() {
  TF_DeleteGraph (graph_);
}

NAN_MODULE_INIT(Graph::Init) {
  Nan::HandleScope scope;
  Local<FunctionTemplate> ctor = Nan::New<FunctionTemplate>(New);
  ctor_.Reset(ctor);

  ctor->SetClassName(class_name());
  ctor->InstanceTemplate()->SetInternalFieldCount(1);
  Local<ObjectTemplate> proto = ctor->PrototypeTemplate();

  SetAccessor(proto, __js("operations"), getOperations);
  SetMethod(proto, "operation", getOperation);
  SetMethod(proto, "hack", hack);

  ctor_instance_.Reset(ctor->GetFunction());
  Nan::Set(target, class_name(), ctor->GetFunction());
}

NAN_METHOD(Graph::New) {
  Nan::HandleScope scope;
  auto o = new Graph();

  if (info.Length() > 0 && node::Buffer::HasInstance(info[0])) {
    TF_Buffer* buf = TF_NewBufferFromString(node::Buffer::Data(info[0]), node::Buffer::Length(info[0]));
    TF_Status* status = TF_NewStatus();

    TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
    if (info.Length() > 1 && info[1]->IsString()) {
      String::Utf8Value prefix(info[1]);
      TF_ImportGraphDefOptionsSetPrefix(opts, *prefix);
    }

    TF_GraphImportGraphDef(o->graph_, buf, opts, status);

    TF_DeleteStatus(status);
    TF_DeleteBuffer(buf);
    TF_DeleteImportGraphDefOptions(opts);
  }

  if (info.IsConstructCall()) {
    o->Wrap(info.This());
    RETURN(info.This());
  }
  else {
    MaybeLocal<Object> instance = Nan::NewInstance(Nan::New(ctor_instance_));
    o->Wrap(instance.ToLocalChecked());
    RETURN(instance.ToLocalChecked());
  }
}

NAN_METHOD(Graph::getOperation) {
  Nan::HandleScope scope;
  auto self = Unwrap<Graph>(info.This());
  String::Utf8Value name(info[0]);

  auto oper = TF_GraphOperationByName(self->graph_, *name);
  if (oper) {
    Local<Object> result = Nan::New<Object>();
    Nan::Set(result, __js("type"), __js(TF_OperationOpType(oper)));
    Nan::Set(result, __js("device"), __js(TF_OperationDevice(oper)));
    Nan::Set(result, __js("numInputs"), __js(TF_OperationNumInputs(oper)));
    Nan::Set(result, __js("numOutputs"), __js(TF_OperationNumOutputs(oper)));
    
    // TODO: list attributes

    RETURN(result);
  }
}

NAN_GETTER(Graph::getOperations) {
  Nan::HandleScope scope;
  auto self = Unwrap<Graph>(info.This());
  Local<Array> result = Nan::New<Array>();
  size_t pos = 0;
  TF_Operation* oper;
  while ((oper = TF_GraphNextOperation(self->graph_, &pos)) != nullptr) {
    result->Set(result->Length(), __js(TF_OperationName(oper)));
  }
  RETURN(result);
}
