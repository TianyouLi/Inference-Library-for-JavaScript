#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

static Nan::Persistent<FunctionTemplate> mat_ctor_p;

static cv::Scalar ArrayToColor(Local<Object> a) {
  Local<Value> b = a->Get(0);
  Local<Value> g = a->Get(1);
  Local<Value> r = a->Get(2);

  return cv::Scalar(b->IntegerValue(), g->IntegerValue(), r->IntegerValue());
}

class JMat: public Nan::ObjectWrap {
public:
  cv::Mat mat_;

  static NAN_MODULE_INIT(Init)
  {
    Nan::HandleScope scope;
    Local<FunctionTemplate> ctor = Nan::New<FunctionTemplate>(JMat::New);
    mat_ctor_p.Reset(ctor);
    ctor->InstanceTemplate()->SetInternalFieldCount(1);
    ctor->SetClassName(__js("Mat"));

    Local<ObjectTemplate> proto = ctor->PrototypeTemplate();
    SetAccessor(proto, __js("cols"), GetCols);
    SetAccessor(proto, __js("rows"), GetRows);

    SetMethod(proto, "rectangle", Rectangle);
  }

  static NAN_METHOD(New) {
    Nan::HandleScope scope;
    auto mat = new JMat;

    mat->Wrap(info.This());
    RETURN(info.This());
  }

  static NAN_GETTER(GetCols) {
    auto self = Unwrap<JMat>(info.This());
    RETURN(__js(self->mat_.cols));
  }

  static NAN_GETTER(GetRows) {
    auto self = Unwrap<JMat>(info.This());
    RETURN(__js(self->mat_.rows));
  }

  static NAN_METHOD(Rectangle) {
    Nan::HandleScope scope;
    auto self = Unwrap<JMat>(info.This());
    int x0 = info[0]->IntegerValue();
    int y0 = info[1]->IntegerValue();
    int x1 = info[2]->IntegerValue();
    int y1 = info[3]->IntegerValue();
    cv::Scalar color(0, 0, 255);
    int thickness = 1;

    if (info.Length() > 4 && info[4]->IsArray()) {
      color = ArrayToColor(info[4]->ToObject());
    }

    if (info.Length() > 5 && info[5]->IntegerValue())
      thickness = info[5]->IntegerValue();

    cv::rectangle(self->mat_, cv::Point(x0, y0), cv::Point(x1, y1), color, thickness);
  }
};
