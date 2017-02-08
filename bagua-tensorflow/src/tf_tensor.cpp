#include "tf.h"
#include "opencv.h"

static TF_DataType dataTypeOf(Local<Value> v) {
  if (v->IsInt32())           return TF_INT32;
  if (v->IsNumber())          return TF_FLOAT; // Maybe TF_DOUBLE
  if (v->IsBoolean())         return TF_BOOL;
  if (v->IsTypedArray()) {
    if (v->IsInt8Array())     return TF_INT8;
    if (v->IsUint8Array())    return TF_STRING;
    if (v->IsInt16Array())    return TF_INT16;
    if (v->IsUint16Array())   return TF_UINT16;
    if (v->IsInt32Array())    return TF_INT32;
    if (v->IsFloat32Array())  return TF_FLOAT;
    if (v->IsFloat64Array())  return TF_DOUBLE;
  }
  else if (v->IsArray()) {
    Local<Array> a = Local<Array>::Cast(v);
    if (a->Length() > 0)      return dataTypeOf(a->Get(0));
  }
  return static_cast<TF_DataType>(0);
}

static int shapeOf(Local<Value> v, std::vector<int64_t>& shape) {
  Nan::HandleScope scope;
  if (v->IsTypedArray()) {
    Local<TypedArray> a = Local<TypedArray>::Cast(v);
    shape.push_back(a->Length());
    return 1;
  }
  if (v->IsArray()) {
    Local<Array> a = Local<Array>::Cast(v);
    if (a->Length() > 0) {
      shape.push_back(a->Length());
      return 1 + shapeOf(a->Get(0), shape);
    }
  }
  return 0;
}

static void jsab_cb (void *data, size_t len, void *arg) {
  auto o = (Tensor *) arg;
  o->ab_p.Reset();
  //printf("%s: %p\n", __func__, o->tensor_);
}

static void jsmat_cb (void *data, size_t len, void *arg) {
  auto o = (Tensor *) arg;
  o->mat_p.Reset();
  //printf("%s: %p %p\n", __func__, o, o->tensor_);
}

static void jsab_dummy (char *data, void *arg) {}

static void writeScalar(Local<Value> v, TF_DataType dtype, void* dst, size_t dst_size) {
  switch (dtype) {
  case TF_DOUBLE:
    *(double *)dst = v->NumberValue();
    break;
  case TF_FLOAT:
    *(float *)dst = v->NumberValue();
    break;
  case TF_INT32:
    *(int32_t *)dst = v->Int32Value();
    break;
  case TF_INT16:
    *(int16_t *)dst = v->Int32Value();
    break;
  case TF_UINT16:
    *(uint16_t *)dst = v->Int32Value();
    break;
  case TF_INT8:
    *(int8_t *)dst = v->Int32Value();
    break;
  case TF_UINT8:
    *(uint8_t *)dst = v->Int32Value();
    break;
  case TF_BOOL:
    *(uint8_t *)dst = v->BooleanValue() ? 1 : 0;
    break;
  default:
    DIE("Unsupport DataType");
  }
}

static size_t write1DArray(Local<Value> v, TF_DataType dtype, void* dst, size_t dst_size) {
  Nan::HandleScope scope;
  if (v->IsTypedArray()) {
    Local<TypedArray> a = Local<TypedArray>::Cast(v);
    Local<ArrayBuffer> ab = a->Buffer();
    auto contents = ab->GetContents();
    if (contents.ByteLength() > dst_size) {
      return 0;
    }
    memcpy (dst, contents.Data(), contents.ByteLength());
    return contents.ByteLength();
  }

  size_t elem_size = TF_DataTypeSize(dtype);
  Local<Array> a = Local<Array>::Cast(v);
  if (a->Length() * elem_size > dst_size) {
    return 0;
  }
  for (size_t i = 0; i < a->Length(); i++) {
    writeScalar(a->Get(i), dtype, (char *)dst + i * elem_size, elem_size);
  }
  return a->Length() * elem_size;
}

static size_t writeNDArray(Local<Value> v, TF_DataType dtype, int num_dims, char* dst, size_t dst_size) {
  if (num_dims == 1) {
    return write1DArray(v, dtype, dst, dst_size);
  }

  Nan::HandleScope scope;
  Local<Array> a = Local<Array>::Cast(v);
  int len = a->Length();
  size_t sz = 0;
  for (int i = 0; i < len; i++) {
    sz += writeNDArray(a->Get(i), dtype, num_dims - 1, dst + sz, dst_size - sz);
  }
  return sz;
}

static void setValue(TF_Tensor *t, Local<Value> v, int num_dims) {
  TF_DataType dtype = TF_TensorType(t);
  void* data = TF_TensorData(t);
  size_t sz = TF_TensorByteSize(t);

  if (num_dims == 0) {
    size_t elem_size = TF_DataTypeSize(dtype);
    if (sz != elem_size) {
      return;
    }
    writeScalar(v, dtype, data, sz);
  }
  else {
    writeNDArray(v, dtype, num_dims, static_cast<char*>(data), sz);
  }
}

IMPL_WRAP(Tensor);

Tensor::Tensor() {
  tensor_ = nullptr;
}

Tensor::~Tensor() {
  if (tensor_ != nullptr) {
    //printf("%s: free tensor %p\n", __func__, tensor_);
    TF_DeleteTensor (tensor_);
  }
}

NAN_MODULE_INIT(Tensor::Init) {
  Nan::HandleScope scope;
  Local<FunctionTemplate> ctor = Nan::New<FunctionTemplate>(New);
  ctor_.Reset(ctor);

  ctor->SetClassName(class_name());
  ctor->InstanceTemplate()->SetInternalFieldCount(1);
  Local<ObjectTemplate> proto = ctor->PrototypeTemplate();

  SetAccessor(proto, __js("type"), type);
  SetAccessor(proto, __js("shape"), shape);
  SetAccessor(proto, __js("data"), getData, setData);

  ctor_instance_.Reset(ctor->GetFunction());
  Nan::Set(target, class_name(), ctor->GetFunction());
}

NAN_METHOD(Tensor::New) {
  Nan::HandleScope scope;
  int argc = info.Length();
  if (argc <= 0) {
    return;
  }

  auto o = new Tensor();
  //printf ("Tensor::New %p...\n", o);

  Local<FunctionTemplate> mat_ctor = Nan::New(Mat::ctor_p);
  if (mat_ctor->HasInstance(info[0])) {
    Local<Object> obj = info[0]->ToObject();
    cv::Mat& mat = Unwrap<Mat>(obj)->mat_;
    int64_t shape[4] = {1, mat.rows, mat.cols, mat.channels()}; // data_format: NHWC
    //printf ("Tensor::New %d,%d,%d,%d...\n", int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]));
    o->mat_p.Reset(obj);
    o->tensor_ = TF_NewTensor(TF_FLOAT, shape, sizeof(shape)/sizeof(shape[0]), mat.data, mat.channels()*mat.rows*mat.cols*sizeof(float), jsmat_cb, o);
  }
  else {
    std::vector<int64_t> shape;
    auto dt = dataTypeOf(info[0]);
    auto elem_size = TF_DataTypeSize(dt);
    int num_elems = 1;

    if (info[0]->IsTypedArray()) {
      Local<TypedArray> a = Local<TypedArray>::Cast(info[0]);
      int num_dims;
      if (argc > 1) {
        num_dims = argc - 1;
        for (int i = 1; i < argc; i++) {
          shape.push_back(info[i]->IntegerValue());
        }
      }
      else {
        num_dims = 1;
        shape.push_back(a->Length());
      }
      for (int i = 0; i < num_dims; i++) {
        num_elems *= shape[i];
      }

      Local<ArrayBuffer> ab = a->Buffer();
      if (num_dims == 1 && info[0]->IsUint8Array()) {
        // TF_STRING tensors are encoded with a table of 8-byte offsets followed by
        // TF_StringEncode-encoded bytes.
        size_t src_len = a->Length();
        size_t dst_len = TF_StringEncodedSize(src_len);
        TF_Tensor* t = TF_AllocateTensor(TF_STRING, nullptr, 0, 8 + dst_len);
        char* dst = static_cast<char*>(TF_TensorData(t));
        memset(dst, 0, 8);  // The offset table

        auto src = ab->GetContents().Data();
        TF_Status* status = TF_NewStatus();
        TF_StringEncode((char *) src, src_len, dst + 8, dst_len, status);
        TF_DeleteStatus(status);
        o->tensor_ = t;
      }
      else {
        o->ab_p.Reset(ab);
        o->tensor_ = TF_NewTensor(dt, &shape[0], num_dims, ab->GetContents().Data(), elem_size * num_elems, jsab_cb, o);
      }
    }
    else {
      int num_dims = shapeOf(info[0], shape);
      for (int i = 0; i < num_dims; i++) {
        num_elems *= shape[i];
      }
      o->tensor_ = TF_AllocateTensor(dt, &shape[0], num_dims, elem_size * num_elems);

      setValue (o->tensor_, info[0], num_dims);
    }
  }
  //printf("%s: New <%d> Tensor => %p\n", __func__, dt, o->tensor_);

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

Local<Object> Tensor::create(TF_Tensor *t) {
  Nan::EscapableHandleScope scope;
  auto o = new Tensor();
  o->tensor_ = t;
  MaybeLocal<Object> instance = Nan::NewInstance(Nan::New(Tensor::ctor_instance_));
  o->Wrap(instance.ToLocalChecked());
  return scope.Escape(instance.ToLocalChecked());
}

NAN_GETTER(Tensor::type) {
  Nan::HandleScope scope;
  auto self = Unwrap<Tensor>(info.This());
  RETURN(Nan::New(int(TF_TensorType(self->tensor_))));
}

NAN_GETTER(Tensor::shape) {
  Nan::HandleScope scope;
  auto self = Unwrap<Tensor>(info.This());
  int num_dims = TF_NumDims(self->tensor_);
  Local<Array> result = Nan::New<Array>(num_dims);
  for (int i = 0; i < num_dims; i++)
    result->Set(i, __js(int(TF_Dim(self->tensor_, i))));
  RETURN(result);
}

NAN_GETTER(Tensor::getData) {
  Nan::HandleScope scope;
  auto self = Unwrap<Tensor>(info.This());
  auto m = TF_TensorData(self->tensor_);
  auto dtype = TF_TensorType(self->tensor_);
  auto num_dims = TF_NumDims(self->tensor_);
  if (num_dims == 0) {
    switch (dtype) {
    case TF_DOUBLE:
      RETURN(__js(*(double *)m));
      break;
    case TF_FLOAT:
      RETURN(__js(*(float *)m));
      break;
    case TF_INT32:
      RETURN(__js(*(int32_t *)m));
      break;
    case TF_INT16:
      RETURN(__js(*(int16_t *)m));
      break;
    case TF_UINT16:
      RETURN(__js(*(uint16_t *)m));
      break;
    case TF_INT8:
      RETURN(__js(*(int8_t *)m));
      break;
    case TF_UINT8:
      RETURN(__js(*(uint8_t *)m));
      break;
    case TF_BOOL:
      RETURN(__js(*(uint8_t *)m == 1));
      break;
    default:
      DIE("Unsupport DataType");
    }
  }
  else {
    auto sz = TF_TensorByteSize(self->tensor_);
    Local<Object> res = Nan::NewBuffer((char *) m, sz, jsab_dummy, self).ToLocalChecked();
    RETURN(res);
  }
}

NAN_SETTER(Tensor::setData) {
  Nan::HandleScope scope;
  auto self = Unwrap<Tensor>(info.This());
  int num_dims = TF_NumDims(self->tensor_);
  setValue (self->tensor_, value, num_dims);
}
