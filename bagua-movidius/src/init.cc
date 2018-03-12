#include <nan.h>
#include <node_object_wrap.h>
#include <node_buffer.h>
#include <libusb.h>
#include "mvnc.h"
#include "fp16.h"

using namespace v8;
using namespace node;


class Device : public Nan::ObjectWrap {
public:
  static void Init(Local<Object> target) {
  	Nan::HandleScope scope;
	Local<FunctionTemplate> tpl = Nan::New<FunctionTemplate>();
	ctor.Reset(tpl);
	tpl->InstanceTemplate()->SetInternalFieldCount(1);
	tpl->SetClassName(Nan::New("Device").ToLocalChecked());

	Nan::SetPrototypeMethod(tpl, "OpenDevice", OpenDevice);
	Nan::SetPrototypeMethod(tpl, "CloseDevice", CloseDevice);
	Nan::SetPrototypeMethod(tpl, "AllocateGraph", AllocateGraph);
	Nan::SetPrototypeMethod(tpl, "SetDeviceOption", SetDeviceOption);
	Nan::SetPrototypeMethod(tpl, "GetDeviceOption", GetDeviceOption);
  }

  template<typename... Args>
  static Local<Object> New(Args... args) {
  	Nan::EscapableHandleScope scope;
  	Local<Object> instance = Nan::NewInstance(Nan::GetFunction(Nan::New(ctor)).ToLocalChecked()).ToLocalChecked();
  	auto self = new Device(args...);
    self->Wrap(instance);
    return scope.Escape(instance);
  }

  static NAN_METHOD(OpenDevice) {
	Nan::HandleScope scope;
	auto self = ObjectWrap::Unwrap<Device>(info.Holder());
	if (self->handle_) {
		mvncCloseDevice(self->handle_);
		self->handle_ = NULL;
	}

  	mvncStatus ret = mvncOpenDevice(self->name_.c_str(), &self->handle_);

  	//fprintf(stderr, "mvncOpenDevice returns %d\n", ret);
  	info.GetReturnValue().Set(ret == MVNC_OK ? Nan::True() : Nan::False());
  }

  static NAN_METHOD(CloseDevice) {
  	Nan::HandleScope scope;
  	auto self = ObjectWrap::Unwrap<Device>(info.Holder());
	if (self->handle_) {
		mvncStatus ret = mvncCloseDevice(self->handle_);
		//fprintf(stderr, "CloseDevice returns %d\n", ret);
		self->handle_ = NULL;
		info.GetReturnValue().Set(ret == MVNC_OK ? Nan::True() : Nan::False());
	}
  }

  static NAN_METHOD(SetDeviceOption) {
	Nan::HandleScope scope;
	auto self = ObjectWrap::Unwrap<Device>(info.Holder());
	//TODO:
  }

  static NAN_METHOD(GetDeviceOption) {
	Nan::HandleScope scope;
	auto self = ObjectWrap::Unwrap<Device>(info.Holder());
	//TODO:
  }

  static NAN_METHOD(AllocateGraph);

  Device(const char *name) : name_(name), handle_(NULL) { fprintf(stderr, "Device %p\n", this); }
  ~Device() { fprintf(stderr, "~Device %p\n", this); }

private:
  static Nan::Persistent<FunctionTemplate> ctor;

  std::string name_;
  void *handle_;
};

class Graph : public Nan::ObjectWrap {
public:
  static void Init(Local<Object> target) {
	Nan::HandleScope scope;
	Local<FunctionTemplate> tpl = Nan::New<FunctionTemplate>();
	ctor.Reset(tpl);
	tpl->InstanceTemplate()->SetInternalFieldCount(1);
	tpl->SetClassName(Nan::New("Graph").ToLocalChecked());

	Nan::SetPrototypeMethod(tpl, "DeallocateGraph", DeallocateGraph);
	Nan::SetPrototypeMethod(tpl, "LoadTensor", LoadTensor);
	Nan::SetPrototypeMethod(tpl, "GetResult", GetResult);
	Nan::SetPrototypeMethod(tpl, "SetGraphOption", SetGraphOption);
	Nan::SetPrototypeMethod(tpl, "GetGraphOption", GetGraphOption);
  }

  template<typename... Args>
  static Local<Object> New(Args... args) {
  	Nan::EscapableHandleScope scope;
  	Local<Object> instance = Nan::NewInstance(Nan::GetFunction(Nan::New(ctor)).ToLocalChecked()).ToLocalChecked();
  	auto self = new Graph(args...);
    self->Wrap(instance);
    return scope.Escape(instance);
  }

  static NAN_METHOD(DeallocateGraph) {
	Nan::HandleScope scope;
	auto self = ObjectWrap::Unwrap<Graph>(info.Holder());
	if (self->handle_) {
	  mvncStatus ret = mvncDeallocateGraph(self->handle_);
	  self->handle_ = NULL;
	  //fprintf(stderr, "DeallocateGraph returns %d\n", ret);
	  info.GetReturnValue().Set(ret == MVNC_OK ? Nan::True() : Nan::False());
	}
  }

  static NAN_METHOD(LoadTensor) {
	Nan::HandleScope scope;
	auto self = ObjectWrap::Unwrap<Graph>(info.Holder());
	if (self->handle_) {
	  if (info.Length() > 0 && info[0]->IsFloat32Array()) {
      	Local<ArrayBuffer> ab = Local<Float32Array>::Cast(info[0])->Buffer();
      	auto contents = ab->GetContents();
      	//fprintf(stderr, " == LoadTensor %d bytes\n", (int) ab->ByteLength());
      	auto imgfp16 = (unsigned char *)malloc(contents.ByteLength() / 2);
      	if (imgfp16) {
      	  floattofp16(imgfp16, (float *)contents.Data(), contents.ByteLength() / 4);
      	  mvncStatus ret = mvncLoadTensor(self->handle_, imgfp16, contents.ByteLength() / 2, NULL);
      	  free(imgfp16);
      	  //fprintf(stderr, "mvncLoadTensor returns %d\n", ret);
      	  info.GetReturnValue().Set(ret == MVNC_OK ? Nan::True() : Nan::False());
      	}
	  }
	}
  }

  static NAN_METHOD(GetResult) {
	Nan::HandleScope scope;
	auto self = ObjectWrap::Unwrap<Graph>(info.Holder());
	if (self->handle_) {
	  void* resultData16;
      void* userParam;
      unsigned int resultBytes;
      mvncStatus ret = mvncGetResult(self->handle_, &resultData16, &resultBytes, &userParam);
      //fprintf(stderr, "mvncGetResult returns %d, %d bytes\n", ret, resultBytes);
      if (ret == MVNC_OK) {
      	Local<ArrayBuffer> ab = ArrayBuffer::New(info.GetIsolate(), resultBytes*2);
      	Local<Float32Array> fp32 = Float32Array::New(ab, 0, resultBytes*2);
      	auto contents = ab->GetContents();
    	fp16tofloat((float*)contents.Data(), (unsigned char*)resultData16, resultBytes/2);
    	info.GetReturnValue().Set(fp32);
      }
	}
  }

  static NAN_METHOD(SetGraphOption) {
	Nan::HandleScope scope;
	auto self = ObjectWrap::Unwrap<Graph>(info.Holder());
	//TODO:
  }

  static NAN_METHOD(GetGraphOption) {
	Nan::HandleScope scope;
	auto self = ObjectWrap::Unwrap<Graph>(info.Holder());
	//TODO:
  }

  Graph(void *handle) : handle_(handle) { fprintf(stderr, "Graph %p\n", this); }
  ~Graph() { fprintf(stderr, "~Graph %p\n", this); }

private:
  static Nan::Persistent<FunctionTemplate> ctor;

  void *handle_;
};

Nan::Persistent<FunctionTemplate> Device::ctor;
Nan::Persistent<FunctionTemplate> Graph::ctor;


NAN_METHOD(Device::AllocateGraph) {
  Nan::HandleScope scope;
  auto self = ObjectWrap::Unwrap<Device>(info.Holder());

  if (info.Length() > 0 && Buffer::HasInstance(info[0])) {
	size_t len = Buffer::Length(info[0]);
	const char* buf = Buffer::Data(info[0]);
	void* gh = NULL;

	mvncStatus ret = mvncAllocateGraph(self->handle_, &gh, buf, len);
	if (ret == MVNC_OK) {
	  Local<Object> obj = Graph::New(gh);
	  info.GetReturnValue().Set(obj);
	}
  }
}

static NAN_METHOD(EnumerateDevices) {
  char name[MVNC_MAX_NAME_SIZE + 1];
  Nan::HandleScope scope;
  Local<Array> names = Nan::New<Array>();

  for (int i = 0; i < 128; i++) {
  	mvncStatus ret = mvncGetDeviceName(i, name, MVNC_MAX_NAME_SIZE);
  	if(ret != MVNC_OK)
  		break;

  	Local<Object> obj = Device::New(name);
    names->Set(i, obj);
  }
  info.GetReturnValue().Set(names);
}

extern "C" void init(Local<Object> target) {
	Nan::Export(target, "EnumerateDevices", EnumerateDevices);

	Device::Init(target);
	Graph::Init(target);
}

NODE_MODULE(ncsdk, init)


#ifdef _WIN32
#include <Windows.h>

extern "C" CRITICAL_SECTION mm;

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
					 )
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
		InitializeCriticalSection(&mm);
		libusb_init(NULL);
		break;
	case DLL_THREAD_ATTACH:
	case DLL_THREAD_DETACH:
		break;
	case DLL_PROCESS_DETACH:
		libusb_exit(NULL);
		break;
	}
	return TRUE;
}
#endif
