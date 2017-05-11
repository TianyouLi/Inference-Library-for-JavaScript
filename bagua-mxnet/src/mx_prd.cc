#include "mx.h"
#include "opencv.h"

typedef unsigned int mx_uint;
typedef float mx_float;
typedef void *PredictorHandle;
typedef void *NDListHandle;

const mx_float DEFAULT_MEAN = 117.0;

void GetImageFile(cv::Mat img, mx_float *image_data, const int channels,
                  const cv::Size resize_size, const mx_float *mean_data = nullptr)
{
    // Read all kinds of file into a BGR color 3 channels image

    cv::Mat im_ori;

    if (img.channels() == 4)
        cv::cvtColor(img, im_ori, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1)
        cv::cvtColor(img, im_ori, cv::COLOR_GRAY2BGR);
    else
        im_ori = img;

    if (im_ori.empty())
    {
        std::cerr << "Can't open the image. Please check. \n";
        assert(false);
    }

    cv::Mat im;

    resize(im_ori, im, resize_size);

    int size = im.rows * im.cols * channels;

    mx_float *ptr_image_r = image_data;
    mx_float *ptr_image_g = image_data + size / 3;
    mx_float *ptr_image_b = image_data + size / 3 * 2;

    float mean_b, mean_g, mean_r;
    mean_b = mean_g = mean_r = DEFAULT_MEAN;

    for (int i = 0; i < im.rows; i++)
    {
        uchar *data = im.ptr<uchar>(i);

        for (int j = 0; j < im.cols; j++)
        {
            if (mean_data)
            {
                mean_r = *mean_data;
                if (channels > 1)
                {
                    mean_g = *(mean_data + size / 3);
                    mean_b = *(mean_data + size / 3 * 2);
                }
                mean_data++;
            }
            if (channels > 1)
            {
                *ptr_image_g++ = static_cast<mx_float>(*data++) - mean_g;
                *ptr_image_b++ = static_cast<mx_float>(*data++) - mean_b;
            }
            *ptr_image_r++ = static_cast<mx_float>(*data++) - mean_r;
            ;
        }
    }
};

void PrintOutputResult(const std::vector<float> &data, const std::vector<std::string> &synset)
{
    if (data.size() != synset.size())
    {
        std::cerr << "Result data and synset size does not match!" << std::endl;
    }

    float best_accuracy = 0.0;
    int best_idx = 0;

    for (int i = 0; i < static_cast<int>(data.size()); i++)
    {
        printf("Accuracy[%d] = %.8f\n", i, data[i]);

        if (data[i] > best_accuracy)
        {
            best_accuracy = data[i];
            best_idx = i;
        }
    }

    printf("Best Result: [%s] id = %d, accuracy = %.8f\n",
           synset[best_idx].c_str(), best_idx, best_accuracy);
}

IMPL_WRAP(MPrd);

MPrd::MPrd()
{
    pred_hnd = 0;
}

NAN_MODULE_INIT(MPrd::Init)
{
    Nan::HandleScope scope;
    Local<FunctionTemplate> ctor = Nan::New<FunctionTemplate>(New);
    ctor_.Reset(ctor);

    ctor->SetClassName(class_name());
    ctor->InstanceTemplate()->SetInternalFieldCount(1);
    Local<ObjectTemplate> proto = ctor->PrototypeTemplate();

    SetMethod(proto, "create", create);
    SetMethod(proto, "setInput", setInput);
    SetMethod(proto, "run", run);
    ctor_instance_.Reset(ctor->GetFunction());
    Nan::Set(target, class_name(), ctor->GetFunction());
    
}

NAN_METHOD(MPrd::New)
{
    Nan::HandleScope scope;
    auto prd = new MPrd();
    prd->Wrap(info.This());
    RETURN(info.This());
}

NAN_METHOD(MPrd::create)
{
    Nan::HandleScope scope;
    auto self = Unwrap<MPrd>(info.This());
    auto params = __cxx_vec<int>(info[0]);
    const char *j = node::Buffer::Data(info[1]);
    const char *p = node::Buffer::Data(info[2]);
    auto l = node::Buffer::Length(info[2]);

    mx_uint num_input_nodes = 1;
    const char *input_key[1] = {"data"};
    const char **input_keys = input_key;
    int width = params[0];
    int height = params[1];
    int channels = params[2];
    int dev_type = 1;
    int dev_id = 0;
    const mx_uint input_shape_indptr[2] = {0, 4};
    const mx_uint input_shape_data[4] = {1,
                                         static_cast<mx_uint>(channels),
                                         static_cast<mx_uint>(width),
                                         static_cast<mx_uint>(height)};
    self->pred_hnd = 0;

    int x = MXPredCreate(j,
                         p,
                         l,
                         dev_type,
                         dev_id,
                         num_input_nodes,
                         input_keys,
                         input_shape_indptr,
                         input_shape_data,
                         &self->pred_hnd);
    assert(self->pred_hnd);

    RETURN(x);
}

// NAN_METHOD(MPrd::readMean)
// {
//     Nan::HandleScope scope;
//     auto self = Unwrap<MPrd>(info.This());

//     // nd_hnd = self->nd_hnd;
//     // nd_data = self->nd_data;

//     // Read Mean Data
//     self->nd_data = NULL;
//     self->nd_hnd = 0;
//     const char *nd_buf = node::Buffer::Data(info[0]);
//     int nd_buf_length = node::Buffer::Length(info[0]);

//     if (nd_buf_length > 0)
//     {
//         mx_uint nd_index = 0;
//         mx_uint nd_len;
//         const mx_uint *nd_shape = 0;
//         const char *nd_key = 0;
//         mx_uint nd_ndim = 0;

//         MXNDListCreate(nd_buf,
//                        nd_buf_length,
//                        &self->nd_hnd, &nd_len);

//         MXNDListGet(self->nd_hnd, nd_index, &nd_key, &self->nd_data, &nd_shape, &nd_ndim);
//     }

//     const mx_float *data;
//     int width = 224;
//     int height = 224;
//     int channels = 3;
//     int image_size = width * height * channels;
// }

NAN_METHOD(MPrd::setInput)
{
    Nan::HandleScope scope;
    auto self = Unwrap<MPrd>(info.This());

    // Read Image Data
    Nan::Utf8String text(info[0]);

    cv::Mat &img = Unwrap<Mat>(info[0]->ToObject())->mat_;
    int width = img.rows;
    int height = img.cols;
    int channels = img.channels();
    int img_size = width * height * channels;
    std::vector<mx_float> image_data = std::vector<mx_float>(img_size);

    GetImageFile(img, image_data.data(),
                 channels, cv::Size(width, height), NULL);

    int x = MXPredSetInput(self->pred_hnd, "data", image_data.data(), img_size);
    RETURN(x);
}

NAN_METHOD(MPrd::run)
{
    Nan::HandleScope scope;
    auto self = Unwrap<MPrd>(info.This());

    // Do Predict Forward
    MXPredForward(self->pred_hnd);

    mx_uint output_index = 0;
    mx_uint *shape = 0;
    mx_uint shape_len;

    // Get Output Result
    MXPredGetOutputShape(self->pred_hnd, output_index, &shape, &shape_len);

    size_t size = 1;
    for (mx_uint i = 0; i < shape_len; ++i)
        size *= shape[i];

    std::vector<float> output(size);

    MXPredGetOutput(self->pred_hnd, output_index, &(output[0]), size);

    // Release NDList
    // if (self->nd_hnd)
    //     MXNDListFree(self->nd_hnd);

    // Release Predictor
    MXPredFree(self->pred_hnd);

    RETURN(__js(output));
}