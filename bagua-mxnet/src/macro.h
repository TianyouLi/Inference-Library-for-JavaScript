

/*! \brief manually define unsigned int */
typedef unsigned int mx_uint;
/*! \brief manually define float */
typedef float mx_float;
/*! \brief handle to Predictor */
typedef void *PredictorHandle;
/*! \brief handle to NDArray list */
typedef void *NDListHandle;

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

#define TOCHAR(t) \
  *v8::String::Utf8Value(t->ToString()) \

#define TOSTR(t) \
    std::string(*(v8::String::Utf8Value)(t->ToString()))    \

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

template <typename T>
inline static Local<Array> __js(const std::vector<T> v){
    Local<Array> res = Nan::New<v8::Array>();
    int i = 0;
    for (auto iter = v.begin(); iter != v.end(); ++iter)
        res->Set(i++, Nan::New(*iter));
    return res;
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

