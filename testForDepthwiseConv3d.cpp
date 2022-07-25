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
  std::vector<int64_t> input_size = {
      1, 1, num_input_size, num_input_size, num_input_size};
  std::vector<int64_t> kernel_size = {2, 1, 1};
  std::vector<int64_t> weight_size = {3, 1, 2, 1, 1};
  at::Tensor input = torch::tensor(
      {{{{{1.7977, 0.5558}, {1.2604, 0.5936}},
         {{-0.1670, 0.1858}, {-0.3711, 0.1215}}}}});
  at::Tensor bias_opt = torch::tensor({0.2575, -1.5157, 1.5102});
  at::Tensor weight = torch::tensor(
      {{{{{-0.6691}}, {{-0.4510}}}},
       {{{{1.2577}}, {{-0.5578}}}},
       {{{{1.4599}}, {{-1.1863}}}}});
  std::vector<int64_t> stride = {1, 1, 1};
  std::vector<int64_t> padding = {1, 1, 1};
  std::vector<int64_t> dilation = {1, 1, 1};

  auto device = torch::kCUDA;
    at::Tensor input_device = input.to(device);
    at::Tensor weight_device = weight.to(device);
    at::Tensor bias_opt_device = bias_opt.to(device);
    // std::cout << "\ninput:\n";
    // std::cout << input << std::endl;
    // std::cout << "\nbias_opt:\n";
    // std::cout << bias_opt << std::endl;
    // std::cout << "\nweight:\n";
    // std::cout << weight << std::endl;

     at::Tensor output = at::native::conv_depthwise3d_cuda(
        input, weight, kernel_size, bias_opt, stride, padding, dilation);

    std::cout << "\noutput:\n";
    std::cout << output;
    std::cout << "torch is available\n";
}