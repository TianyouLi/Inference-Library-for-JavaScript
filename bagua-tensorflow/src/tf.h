#include <node.h>
#include <nan.h>

#include <string.h>
#include <sstream>
#include <vector>
#include <list>
#include <memory>

#include <tensorflow/c/c_api.h>

#define DIE(msg)  do { fputs (msg, stderr); abort(); } while(0)

#define ARG(i)    info.Length() > 0 && i < info.Length() ? info[i] : Local<Value>(Nan::Undefined())
#define RETURN(x) info.GetReturnValue().Set(x)

#define DECLARE_WRAP(t) \
  static Local<String> class_name() { return Nan::New(#t).ToLocalChecked(); } \
  static Nan::Persistent<FunctionTemplate> ctor_; \
  static Nan::Persistent<Function> ctor_instance_

#define IMPL_WRAP(t) \
Nan::Persistent<FunctionTemplate> t::ctor_; \
Nan::Persistent<Function> t::ctor_instance_

using namespace v8;

// Cast to JS types
inline static Local<Boolean> __js (bool b) {
  return Nan::New(b);
}

inline static Local<Number> __js (int i) {
  return Nan::New(i);
}

inline static Local<Number> __js (double r) {
  return Nan::New(r);
}

inline static Local<String> __js (const char *str) {
  return Nan::New(str).ToLocalChecked();
}

inline static Local<String> __js (const std::string &str) {
  return __js(str.c_str());
}


// Convert to C++ types
inline static void __cxx (Local<Value> val, int& i) {
  i = val->IntegerValue();
}

template <typename T>
inline static void __cxx (Local<Value> val, T& f) {
  f = val->NumberValue();
}

template <typename T>
inline static const std::vector<T> __cxx_vec (Local<Value> v) {
  std::vector<T> vec;
  Local<Object> obj;
  size_t len = 0;

  if (v->IsObject()) {
    obj = v->ToObject();
    if (v->IsArray()) {
      len = Local<Array>::Cast(v)->Length();
    }
    else {
      Nan::MaybeLocal<Value> prop = Nan::Get(obj, __js("Length"));
      if (!prop.IsEmpty()) {
        Nan::MaybeLocal<Uint32> length = Nan::ToArrayIndex(prop.ToLocalChecked());
        if (!length.IsEmpty() && length.ToLocalChecked()->IsUint32())
          len = length.ToLocalChecked()->Uint32Value();
      }
    }
  }
  if (len) {
    for (size_t i = 0; i < len; i++) {
      T val;
      __cxx((*obj)->Get(i), val);
      vec.push_back(val);
    }
  }
  return vec;
}

// Tensorflow
class DataType {
public:
  static NAN_MODULE_INIT(Init);
};


class Tensor : public Nan::ObjectWrap {
public:
  Tensor();
  virtual ~Tensor();

  static NAN_MODULE_INIT(Init);
  static NAN_METHOD(New);
  static NAN_GETTER(type);
  static NAN_GETTER(shape);
  static NAN_GETTER(getData);
  static NAN_SETTER(setData);

  static Local<Object> create(TF_Tensor *);

  TF_Tensor*  tensor_;
  Nan::Persistent<ArrayBuffer> ab_p;
  Nan::Persistent<Object> mat_p;

  DECLARE_WRAP(Tensor);
};


class Graph : public Nan::ObjectWrap {
public:
  Graph();
  virtual ~Graph();

  static NAN_MODULE_INIT(Init);
  static NAN_METHOD(New);
  static NAN_METHOD(getOperation);
  static NAN_GETTER(getOperations);

  TF_Graph* graph_;

  DECLARE_WRAP(Graph);
};


class Session : public Nan::ObjectWrap {
public:
  Session(TF_Session*);
  virtual ~Session();

  static NAN_MODULE_INIT(Init);
  static NAN_METHOD(New);
  static NAN_METHOD(run);
  static NAN_METHOD(close);

  TF_Graph* graph_;
  TF_Session* session_;

  DECLARE_WRAP(Session);
};
