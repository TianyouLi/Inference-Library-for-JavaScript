#include <algorithm>
#include <vector>

#include "caffe/bn_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  const Dtype* const_bottom_data = bottom[0]->gpu_data();
  const Dtype* const_top_data = top[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  const Dtype* scale_data = this->blobs_[0]->gpu_data();
  const Dtype* shift_data = this->blobs_[1]->gpu_data();

if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
  // Mean normalization
  if (frozen_ || this->phase_ == TEST) {
    // Use the moving average mean
    caffe_copy(batch_statistic_.count(), this->blobs_[2]->gpu_data(),
        batch_statistic_.mutable_gpu_data());
  } else {
    // Compute the mean by averaging over spatial and batch dimensions.
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Dtype(1) / (height_ * width_), const_bottom_data,
        spatial_sum_multiplier_.gpu_data(), Dtype(0),
        spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_,
        Dtype(1) / num_, spatial_statistic_.gpu_data(),
        batch_sum_multiplier_.gpu_data(), Dtype(0),
        batch_statistic_.mutable_gpu_data());
    // Add to the moving average
    if (!frozen_) {
      caffe_gpu_axpby(batch_statistic_.count(),
          Dtype(1) - bn_momentum_, batch_statistic_.gpu_data(),
          bn_momentum_, this->blobs_[2]->mutable_gpu_data());
    }
  }
  // Broadcast the mean vector
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Dtype(1), batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(),
      Dtype(0), spatial_statistic_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Dtype(-1),
      spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
      Dtype(0), broadcast_buffer_.mutable_gpu_data());
  // Subtract
  caffe_gpu_add(broadcast_buffer_.count(), const_bottom_data,
      broadcast_buffer_.gpu_data(), top_data);

  // Variance normalization
  if (frozen_ || this->phase_ == TEST) {
    // Use the moving average variance
    caffe_copy(batch_statistic_.count(), this->blobs_[3]->gpu_data(),
        batch_statistic_.mutable_gpu_data());
  } else {
    caffe_gpu_powx(broadcast_buffer_.count(), const_top_data, Dtype(2),
        broadcast_buffer_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Dtype(1) / (height_ * width_), broadcast_buffer_.gpu_data(),
        spatial_sum_multiplier_.gpu_data(), Dtype(0),
        spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1) / num_,
        spatial_statistic_.gpu_data(), batch_sum_multiplier_.gpu_data(),
        Dtype(0), batch_statistic_.mutable_gpu_data());

    // Add to the moving average
    caffe_gpu_axpby(batch_statistic_.count(),
        Dtype(1) - bn_momentum_, batch_statistic_.gpu_data(),
        bn_momentum_, this->blobs_[3]->mutable_gpu_data());
  }

  // Add eps
  caffe_gpu_add_scalar(batch_statistic_.count(), bn_eps_,
        batch_statistic_.mutable_gpu_data());
  // Inverse standard deviation
  caffe_gpu_powx(batch_statistic_.count(), batch_statistic_.gpu_data(),
        Dtype(-0.5), batch_statistic_.mutable_gpu_data());
  // Broadcast the inverse std
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Dtype(1), batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(),
      Dtype(0), spatial_statistic_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Dtype(1),
      spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
      Dtype(0), broadcast_buffer_.mutable_gpu_data());
  // Multiply with the inverse std
  caffe_gpu_mul(broadcast_buffer_.count(), const_top_data,
      broadcast_buffer_.gpu_data(), top_data);

  // Save the normalized inputs and std for backprop
  if (!frozen_) {
    caffe_copy(broadcast_buffer_.count(), const_top_data,
        x_norm_.mutable_gpu_data());
    caffe_copy(batch_statistic_.count(), batch_statistic_.gpu_data(),
        x_inv_std_.mutable_gpu_data());
  }

  // Scale
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Dtype(1), batch_sum_multiplier_.gpu_data(), scale_data,
      Dtype(0), spatial_statistic_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Dtype(1),
      spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
      Dtype(0), broadcast_buffer_.mutable_gpu_data());
  caffe_gpu_mul(broadcast_buffer_.count(), const_top_data,
      broadcast_buffer_.gpu_data(), top_data);

  // Shift
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Dtype(1), batch_sum_multiplier_.gpu_data(), shift_data,
      Dtype(0), spatial_statistic_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Dtype(1),
      spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
      Dtype(0), broadcast_buffer_.mutable_gpu_data());
  caffe_gpu_add(broadcast_buffer_.count(), const_top_data,
      broadcast_buffer_.gpu_data(), top_data);
#endif // USE_CUDA
} else {
#ifdef USE_GREENTEA
  int_tp devid = this->device_->id();
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(devid);
  // Mean normalization
  if (frozen_ || this->phase_ == TEST) {
    // Use the moving average mean
    greentea_copy<Dtype>(batch_statistic_.count(), (cl_mem) this->blobs_[2]->gpu_data(), 0,
        (cl_mem) batch_statistic_.mutable_gpu_data(), 0, &ctx);
  } else {
    // Compute the mean by averaging over spatial and batch dimensions.
    /*caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Dtype(1) / (height_ * width_), const_bottom_data,
        spatial_sum_multiplier_.gpu_data(), Dtype(0),
        spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_,
        Dtype(1) / num_, spatial_statistic_.gpu_data(),
        batch_sum_multiplier_.gpu_data(), Dtype(0),
        batch_statistic_.mutable_gpu_data());
    // Add to the moving average
    if (!frozen_) {
      caffe_gpu_axpby(batch_statistic_.count(),
          Dtype(1) - bn_momentum_, batch_statistic_.gpu_data(),
          bn_momentum_, this->blobs_[2]->mutable_gpu_data());
    }*/
    LOG(FATAL)<< __FILE__ << ": " << __LINE__ << ":Unimplemented, Inference Only!";
  }
  // Broadcast the mean vector
  greentea_gpu_gemm<Dtype>(devid, CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Dtype(1), (cl_mem) batch_sum_multiplier_.gpu_data(), 0, (cl_mem) batch_statistic_.gpu_data(), 0,
      Dtype(0), (cl_mem) spatial_statistic_.mutable_gpu_data(), 0);
  greentea_gpu_gemm<Dtype>(devid, CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Dtype(-1),
      (cl_mem) spatial_statistic_.gpu_data(), 0, (cl_mem) spatial_sum_multiplier_.gpu_data(), 0,
      Dtype(0), (cl_mem) broadcast_buffer_.mutable_gpu_data(), 0);
  // Subtract
  greentea_gpu_add<Dtype>(devid, broadcast_buffer_.count(), (cl_mem) const_bottom_data, 0,
      (cl_mem) broadcast_buffer_.gpu_data(), 0, (cl_mem) top_data, 0);

  // Variance normalization
  if (frozen_ || this->phase_ == TEST) {
    // Use the moving average variance
    greentea_copy<Dtype>(batch_statistic_.count(), (cl_mem) this->blobs_[3]->gpu_data(), 0,
        (cl_mem) batch_statistic_.mutable_gpu_data(), 0, &ctx);
  } else {
    /*caffe_gpu_powx(broadcast_buffer_.count(), const_top_data, Dtype(2),
        broadcast_buffer_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Dtype(1) / (height_ * width_), broadcast_buffer_.gpu_data(),
        spatial_sum_multiplier_.gpu_data(), Dtype(0),
        spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1) / num_,
        spatial_statistic_.gpu_data(), batch_sum_multiplier_.gpu_data(),
        Dtype(0), batch_statistic_.mutable_gpu_data());

    // Add to the moving average
    caffe_gpu_axpby(batch_statistic_.count(),
        Dtype(1) - bn_momentum_, batch_statistic_.gpu_data(),
        bn_momentum_, this->blobs_[3]->mutable_gpu_data());
    */
    LOG(FATAL)<< __FILE__ << ": " << __LINE__ << ":Unimplemented, Inference Only!";
  }

  // Add eps
  greentea_gpu_add_scalar(devid, batch_statistic_.count(), bn_eps_,
        (cl_mem) batch_statistic_.mutable_gpu_data(), 0);
  // Inverse standard deviation
  greentea_gpu_powx(devid, batch_statistic_.count(), (cl_mem) batch_statistic_.gpu_data(), 0,
        Dtype(-0.5), (cl_mem) batch_statistic_.mutable_gpu_data(), 0);
  // Broadcast the inverse std
  greentea_gpu_gemm<Dtype>(devid, CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Dtype(1), (cl_mem) batch_sum_multiplier_.gpu_data(), 0, (cl_mem) batch_statistic_.gpu_data(), 0,
      Dtype(0), (cl_mem) spatial_statistic_.mutable_gpu_data(), 0);
  greentea_gpu_gemm<Dtype>(devid, CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Dtype(1),
      (cl_mem) spatial_statistic_.gpu_data(), 0, (cl_mem) spatial_sum_multiplier_.gpu_data(), 0,
      Dtype(0), (cl_mem) broadcast_buffer_.mutable_gpu_data(), 0);
  // Multiply with the inverse std
  greentea_gpu_mul<Dtype>(devid, broadcast_buffer_.count(), (cl_mem) const_top_data, 0,
      (cl_mem) broadcast_buffer_.gpu_data(), 0, (cl_mem) top_data, 0);

  // Save the normalized inputs and std for backprop
  if (!frozen_) {
    /*caffe_copy(broadcast_buffer_.count(), const_top_data,
        x_norm_.mutable_gpu_data());
    caffe_copy(batch_statistic_.count(), batch_statistic_.gpu_data(),
        x_inv_std_.mutable_gpu_data());*/
    LOG(FATAL)<< __FILE__ << ": " << __LINE__ << ":Unimplemented, Inference Only!";
  }

  // Scale
  greentea_gpu_gemm<Dtype>(devid, CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Dtype(1), (cl_mem) batch_sum_multiplier_.gpu_data(), 0, (cl_mem) scale_data, 0,
      Dtype(0), (cl_mem) spatial_statistic_.mutable_gpu_data(), 0);
  greentea_gpu_gemm<Dtype>(devid, CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Dtype(1),
      (cl_mem) spatial_statistic_.gpu_data(), 0, (cl_mem) spatial_sum_multiplier_.gpu_data(), 0,
      Dtype(0), (cl_mem) broadcast_buffer_.mutable_gpu_data(), 0);
  greentea_gpu_mul<Dtype>(devid, broadcast_buffer_.count(), (cl_mem) const_top_data, 0,
      (cl_mem) broadcast_buffer_.gpu_data(), 0, (cl_mem) top_data, 0);

  // Shift
  greentea_gpu_gemm<Dtype>(devid, CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Dtype(1), (cl_mem) batch_sum_multiplier_.gpu_data(), 0, (cl_mem) shift_data, 0,
      Dtype(0), (cl_mem) spatial_statistic_.mutable_gpu_data(), 0);
  greentea_gpu_gemm<Dtype>(devid, CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Dtype(1),
      (cl_mem) spatial_statistic_.gpu_data(), 0, (cl_mem) spatial_sum_multiplier_.gpu_data(), 0,
      Dtype(0), (cl_mem) broadcast_buffer_.mutable_gpu_data(), 0);
  greentea_gpu_add<Dtype>(devid, broadcast_buffer_.count(), (cl_mem) const_top_data, 0,
      (cl_mem) broadcast_buffer_.gpu_data(), 0, (cl_mem) top_data, 0);
#endif
}
}

template <typename Dtype>
void BNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
  if (frozen_) {
    if (propagate_down[0]) {
      const Dtype* const_top_diff = top[0]->gpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      // Use the moving average variance
      caffe_copy(batch_statistic_.count(), this->blobs_[3]->gpu_data(),
          batch_statistic_.mutable_gpu_data());
      caffe_gpu_add_scalar(batch_statistic_.count(), bn_eps_,
          batch_statistic_.mutable_gpu_data());
      caffe_gpu_powx(batch_statistic_.count(), batch_statistic_.gpu_data(),
          Dtype(-0.5), batch_statistic_.mutable_gpu_data());
      // Multiple slope with inverse std
      caffe_gpu_mul(batch_statistic_.count(), this->blobs_[0]->gpu_data(),
          batch_statistic_.gpu_data(), batch_statistic_.mutable_gpu_data());
      // Broadcast
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
          Dtype(1), batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(),
          Dtype(0), spatial_statistic_.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
          height_ * width_, 1, Dtype(1),
          spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
          Dtype(0), broadcast_buffer_.mutable_gpu_data());
      // Elementwise multiply top grad with (slope / std)
      caffe_gpu_mul(broadcast_buffer_.count(), const_top_diff,
          broadcast_buffer_.gpu_data(), bottom_diff);
    }
    return;
  }

  // gradient w.r.t. slope
  if (this->param_propagate_down_[0]) {
    const Dtype* const_top_diff = top[0]->gpu_diff();
    Dtype* scale_diff = this->blobs_[0]->mutable_gpu_diff();
    caffe_gpu_mul(broadcast_buffer_.count(), x_norm_.gpu_data(), const_top_diff,
        broadcast_buffer_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Dtype(1), broadcast_buffer_.gpu_data(),
        spatial_sum_multiplier_.gpu_data(), Dtype(0),
        spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
        spatial_statistic_.gpu_data(), batch_sum_multiplier_.gpu_data(),
        Dtype(1), scale_diff);
  }

  // gradient w.r.t. bias
  if (this->param_propagate_down_[1]) {
    const Dtype* const_top_diff = top[0]->gpu_diff();
    Dtype* shift_diff = this->blobs_[1]->mutable_gpu_diff();
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Dtype(1), const_top_diff, spatial_sum_multiplier_.gpu_data(),
        Dtype(0), spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
        spatial_statistic_.gpu_data(), batch_sum_multiplier_.gpu_data(),
        Dtype(1), shift_diff);
  }

  // gradient w.r.t. normalized inputs
  if (propagate_down[0]) {
    const Dtype* const_top_diff = top[0]->gpu_diff();
    const Dtype* const_bottom_diff = bottom[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* scale_data = this->blobs_[0]->gpu_data();
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
        Dtype(1), batch_sum_multiplier_.gpu_data(), scale_data,
        Dtype(0), spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
        height_ * width_, 1, Dtype(1), spatial_statistic_.gpu_data(),
        spatial_sum_multiplier_.gpu_data(), Dtype(0),
        broadcast_buffer_.mutable_gpu_data());
    caffe_gpu_mul(broadcast_buffer_.count(), const_top_diff,
        broadcast_buffer_.gpu_data(), broadcast_buffer_.mutable_gpu_data());

    // sum of x_hat * (dl / dx_hat)
    caffe_gpu_mul(broadcast_buffer_.count(), x_norm_.gpu_data(),
        broadcast_buffer_.gpu_data(), bottom_diff);
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Dtype(1), const_bottom_diff, spatial_sum_multiplier_.gpu_data(),
        Dtype(0), spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
        spatial_statistic_.gpu_data(), batch_sum_multiplier_.gpu_data(),
        Dtype(0), batch_statistic_.mutable_gpu_data());

    // x_hat times the sum
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
        Dtype(1), batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(),
        Dtype(0), spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
        height_ * width_, 1, Dtype(1),
        spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
        Dtype(0), bottom_diff);
    caffe_gpu_mul(broadcast_buffer_.count(), x_norm_.gpu_data(),
        const_bottom_diff, bottom_diff);

    // Subtract the average of x_hat times the sum
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Dtype(1), broadcast_buffer_.gpu_data(),
        spatial_sum_multiplier_.gpu_data(), Dtype(0),
        spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
        spatial_statistic_.gpu_data(), batch_sum_multiplier_.gpu_data(),
        Dtype(0), batch_statistic_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
        Dtype(1), batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(),
        Dtype(0), spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
        height_ * width_, 1, Dtype(1),
        spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
        Dtype(1), bottom_diff);
    caffe_gpu_axpby(broadcast_buffer_.count(), Dtype(1),
        broadcast_buffer_.gpu_data(), Dtype(-1) / (num_ * height_ * width_),
        bottom_diff);

    // Multiply with the inverse std
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
        Dtype(1), batch_sum_multiplier_.gpu_data(), x_inv_std_.gpu_data(),
        Dtype(0), spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
        height_ * width_, 1, Dtype(1),
        spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
        Dtype(0), broadcast_buffer_.mutable_gpu_data());
    caffe_gpu_mul(broadcast_buffer_.count(), const_bottom_diff,
        broadcast_buffer_.gpu_data(), bottom_diff);
  }
#endif // USE_CUDA
} else {
#ifdef USE_GREENTEA
  LOG(FATAL)<< __FILE__ << ": " << __LINE__ << ":Unimplemented, Inference Only!";
#endif
}
}

INSTANTIATE_LAYER_GPU_FUNCS(BNLayer);

}  // namespace caffe
