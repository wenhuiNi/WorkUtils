#include <torch/torch.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorMeta.h>
#include <iostream>

int main()
{
  int num_input_size = 2;
  std::vector<int64_t> input_size = {num_input_size, num_input_size, num_input_size, num_input_size};
  std::vector<int64_t> kernel_size = {2, 2};
  std::vector<int64_t> weight_size = {2 * num_input_size, 1, 3, 3};
  // at::Tensor input = torch::tensor(
  //     {{{{1, 1}, {1, 1}},
  //       {{1, 1}, {1, 1}}},
  //      {{{1, 1}, {1, 1}},
  //       {{1, 1}, {1, 1}}}});
  // at::Tensor weight = torch::tensor(
  //     {{{{1, 1, 1},
  //        {1, 1, 1},
  //        {1, 1, 1}}},
  //      {{{1, 1, 1},
  //        {1, 1, 1},
  //        {1, 1, 1}}},
  //      {{{1, 1, 1},
  //        {1, 1, 1},
  //        {1, 1, 1}}},
  //      {{{1, 1, 1},
  //        {1, 1, 1},
  //        {1, 1, 1}}}});
  // at::Tensor grad_output = torch::tensor(
  //     {{{{1}}, {{1}}, {{1}}, {{1}}},
  //      {{{1}}, {{1}}, {{1}}, {{1}}}});

  at::Tensor input = torch::tensor(
      {{{{0.1245, -0.2392},
         {2.6856, 0.4762}},
        {{0.2791, -1.5399},
         {0.8367, 0.1065}}},
       {{{-0.9192, 0.2802},
         {-0.8946, -0.8632}},
        {{0.9858, 0.6438},
         {-0.0123, -1.1046}}}});
  at::Tensor weight = torch::tensor(
      {{{{-0.3467, 0.4935, -0.7021},
         {1.3579, -1.6696, 2.2722},
         {0.3008, 0.8931, -0.6728}}},
       {{{0.5392, 1.4424, -1.0112},
         {0.6288, -1.5211, -0.2423},
         {-1.2451, 0.9411, -0.3453}}},
       {{{0.3625, 0.3024, 1.6176},
         {-2.1552, 2.0845, 0.0129},
         {-0.1828, 0.5020, 0.4049}}},
       {{{-1.3248, 0.4579, 2.8963},
         {-0.5625, 1.9471, 0.1026},
         {0.1854, 0.1536, -0.1353}}}});
  at::Tensor grad_output = torch::tensor(
      {{{{-0.1538}}, {{0.01 * -2.7718}}, {{-0.7707}}, {{-0.2949}}},
       {{{-0.9664}}, {{-0.6942}}, {{-1.3010}}, {{-1.1833}}}});

  std::vector<int64_t> stride = {2, 2};
  std::vector<int64_t> padding = {2, 2};
  std::vector<int64_t> dilation = {2, 2};

  std::array<bool, 2> output_mask = {true, true};

  if (torch::cuda::is_available())
  {

    auto device = torch::kCUDA;
    at::Tensor input_device = input.to(device);
    at::Tensor weight_device = weight.to(device);

    at::Tensor grad_output_device = grad_output.to(device);
    std::cout << "\ninput:\n";
    std::cout << input << std::endl;
    std::cout << "\ngrad_output:\n";
    std::cout << grad_output << std::endl;
    std::cout << "\nweight:\n";
    std::cout << weight << std::endl;

    std::tuple<at::Tensor, at::Tensor> output_at = at::native::conv_depthwise2d_backward_cuda(
        grad_output_device, input_device, weight_device, kernel_size, stride, padding, dilation, output_mask);

    std::cout << "\noutput:\n";
    std::cout << std::get<0>(output_at).cpu() << std::endl;
    std::cout << std::get<1>(output_at).cpu() << std::endl;
    std::cout << "torch is available\n";
  }
  else
  {
    // auto device = torch::kCPU;
    // at::Tensor output = at::_conv_depthwise2d(
    //     input, weight, kernel_size, bias_opt, stride, padding, dilation);
    std::cout << "cuda is not available\n";
  }
  // at::Tensor output_ref = torch::tensor(
  //     {{{{1.2294, 1.2294}, {1.2294, 1.1862}},
  //       {{-0.6155, -0.6155}, {-0.6155, -0.5541}},
  //       {{-6.6000 * 0.01, -6.6000 * 0.01}, {-6.6000 * 0.01, -26.1956 * 0.01}},
  //       {{-0.2408, -0.2408}, {-0.2408, 0.1382}}},
  //      {{{1.2294, 1.2294}, {1.2294, 1.5481}},
  //       {{-0.6155, -0.6155}, {-0.6155, -1.0691}},
  //       {{-6.6000 * 0.01, -6.6000 * 0.01}, {-6.6000 * 0.01, -75.8130 * 0.01}},
  //       {{-0.2408, -0.2408}, {-0.2408, 1.0978}}}});
  // std::cout << output_ref << std::endl;
  // std::cout << "\ninput:\n";
  // std::cout << input << std::endl;
  // std::cout << "\ngrad_output:\n";
  // std::cout << grad_output << std::endl;
  // std::cout << "\nweight:\n";
  // std::cout << weight << std::endl;
  // std::cout << "\nbias_opt:\n";
  // std::cout << bias_opt << std::endl;
}

// output:
// (1,1,.,.) =
//   1.2294  1.2294
//   1.2294  1.1863

// (2,1,.,.) =
//   1.2294  1.2294
//   1.2294  1.5481

// (1,2,.,.) =
//  -0.6155 -0.6155
//  -0.6155 -0.5541

// (2,2,.,.) =
//  -0.6155 -0.6155
//  -0.6155 -1.0691

// (1,3,.,.) =
//  0.01 *
//  -6.5990 -6.5990
//   -6.5990 -26.1948

// (2,3,.,.) =
//  0.01 *
//  -6.5990 -6.5990
//   -6.5990 -75.8126

// (1,4,.,.) =
//  -0.2408 -0.2408
//  -0.2408  0.1382

// (2,4,.,.) =
//  -0.2408 -0.2408
//  -0.2408  1.0978
