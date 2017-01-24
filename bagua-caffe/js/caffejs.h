#include <node.h>
#include <nan.h>

#include <string.h>
#include <sstream>
#include <vector>
#include <list>

#include <caffe/common.hpp>
#include <caffe/blob.hpp>
#include <caffe/layer.hpp>
#include <caffe/layers/base_data_layer.hpp>
#include <caffe/net.hpp>
#include <caffe/util/io.hpp>
#include <caffe/parallel.hpp>

#define ARG(i)    info.Length() > 0 && i < info.Length() ? info[i] : Local<Value>(Nan::Undefined())
#define RETURN(x) info.GetReturnValue().Set(x)

#define DECLARE_WRAP(t, n) public: \
  template <typename P> static Local<Value> Create(P n) { \
    Nan::EscapableHandleScope scope; \
    Nan::MaybeLocal<Object> instance = Nan::NewInstance(Nan::New(ctor_instance_)); \
    J##t* o = new J##t(n); \
    o->Wrap(instance.ToLocalChecked()); \
    return scope.Escape(instance.ToLocalChecked()); \
  } \
  static Local<String> class_name() { return __js(#t); } \
  static Nan::Persistent<FunctionTemplate> ctor_; \
  static Nan::Persistent<Function> ctor_instance_; \
  shared_ptr<t<Dtype>> n##_; \
}; \
template <typename T> Nan::Persistent<FunctionTemplate> J##t<T>::ctor_; \
template <typename T> Nan::Persistent<Function> J##t<T>::ctor_instance_; \
template <typename Dtype> static Local<Value> __js (const shared_ptr<t<Dtype>> &x) { return J##t<Dtype>::Create(x); }

using namespace v8;
using namespace boost;
using namespace caffe;

// Cast to JS types
static Local<Boolean> __js (bool b) {
  return Nan::New(b);
}

static Local<Number> __js (int i) {
  return Nan::New(i);
}

static Local<String> __js (const char *str) {
  return Nan::New(str).ToLocalChecked();
}

static Local<String> __js (const std::string &str) {
  return __js(str.c_str());
}

template <typename Dtype>
static Local<Value> __js (Blob<Dtype> *);

template <typename Dtype>
static Local<Value> __js (const shared_ptr<Blob<Dtype>> &);

template <typename Dtype>
static Local<Value> __js (const shared_ptr<Net<Dtype>> &);

template <typename T>
static Local<Array> __js_vec (const T *data, size_t n) {
  Local<Array> result = Nan::New<Array>(n);
  for (size_t i = 0; i < n; i++)
    result->Set(i, __js(data[i]));
  return result;
}

template <typename T>
static Local<Array> __js_vec (const std::vector<T>& vec) {
  return __js_vec(&vec[0], vec.size());
}

// Convert to C++ types
static void __cxx (Local<Value> val, int& i) {
  i = val->IntegerValue();
}

template <typename T>
static void __cxx (Local<Value> val, T& f) {
  f = val->NumberValue();
}

template <typename T>
static const std::vector<T> __cxx_vec (Local<Value> v) {
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
