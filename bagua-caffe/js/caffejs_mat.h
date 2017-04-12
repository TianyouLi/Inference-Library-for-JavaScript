#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

static Nan::Persistent<FunctionTemplate> mat_ctor_p;
static Nan::Persistent<Function> mat_ctor_instance_p;

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
    ctor->InstanceTemplate()->SetInternalFieldCount(1);
    ctor->SetClassName(__js("Mat"));

    Local<ObjectTemplate> proto = ctor->PrototypeTemplate();
    SetAccessor(proto, __js("cols"), GetCols);
    SetAccessor(proto, __js("rows"), GetRows);

    SetMethod(proto, "rectangle", Rectangle);
    SetMethod(proto, "putText", putText);
    SetMethod(proto, "fillRect", fillRect);
    SetMethod(proto, "fillPoly", fillPoly);

    mat_ctor_p.Reset(ctor);
    mat_ctor_instance_p.Reset(ctor->GetFunction());
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

  static NAN_METHOD(putText) {
    Nan::HandleScope scope;
    auto self = Unwrap<JMat>(info.This());
    Nan::Utf8String text(info[0]);
    int x = info[1]->IntegerValue();
    int y = info[2]->IntegerValue();
    cv::Scalar color(0, 0, 255);

    if (info.Length() >= 4 && info[3]->IsArray()) {
      color = ArrayToColor(info[3]->ToObject());
    }

    double scale = info.Length() >= 5 ? info[4]->NumberValue() : 1;
    double thickness = info.Length() >= 6 ? info[5]->NumberValue() : 1;

    cv::putText(self->mat_, *text, cv::Point(x, y), cv::FONT_HERSHEY_COMPLEX_SMALL, scale, color, thickness);
  }

  static NAN_METHOD(fillRect) {
    Nan::HandleScope scope;
    auto self = Unwrap<JMat>(info.This());
    int x0 = info[0]->IntegerValue();
    int y0 = info[1]->IntegerValue();
    int x1 = info[2]->IntegerValue();
    int y1 = info[3]->IntegerValue();

    const cv::Point polygon0[4] = {{x0, y0}, {x0, y1}, {x1, y1}, {x1, y0}};
    const cv::Point *polygons[1] = {polygon0};
    const int polySizes[] = {4};

    cv::Scalar color(0, 0, 255);
    if (info.Length() >= 5 && info[4]->IsArray()) {
      color = ArrayToColor(info[4]->ToObject());
    }

    cv::fillPoly(self->mat_, polygons, polySizes, 1, color);
    cv::rectangle(self->mat_, cv::Point(x0, y0), cv::Point(x1, y1), color, 1);
  }

  static NAN_METHOD(fillPoly) {
    Nan::HandleScope scope;
    auto self = Unwrap<JMat>(info.This());
    if (info[0]->IsArray()) {
      Local<Array> polyArray = Local<Array>::Cast(info[0]->ToObject());

      cv::Point **polygons = new cv::Point* [polyArray->Length()];
      int *polySizes = new int[polyArray->Length()];
      for (unsigned int i = 0; i < polyArray->Length(); i++) {
        Local<Array> singlePoly = Local<Array>::Cast(polyArray->Get(i)->ToObject());
        polygons[i] = new cv::Point [singlePoly->Length()];
        polySizes[i] = singlePoly->Length();

        for (unsigned int j = 0; j < singlePoly->Length(); j++) {
          Local<Array> point = Local<Array>::Cast(singlePoly->Get(j)->ToObject());
          polygons[i][j].x = point->Get(0)->IntegerValue();
          polygons[i][j].y = point->Get(1)->IntegerValue();
        }
      }

      cv::Scalar color(0, 0, 255);
      if (info[1]->IsArray()) {
        color = ArrayToColor(info[1]->ToObject());
      }

      cv::fillPoly(self->mat_, (const cv::Point **) polygons, polySizes, polyArray->Length(), color);

      for (unsigned int i = 0; i < polyArray->Length(); i++)
        delete [] polygons[i];
      delete [] polygons;
      delete [] polySizes;
    }
  }
};
