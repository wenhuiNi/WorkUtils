#include <c10/util/Optional.h>
#include <torch/torch.h>
#include <tuple>

int main()
{
  auto device = torch::kCPU;
  std::vector<int64_t> input_size = {2, 2, 2, 2};
  std::vector<int64_t> kernel_size = {2, 2, 2};
  std::vector<int64_t> weight_size = {3,2, 2, 2, 2};
    //at::Tensor input = torch::randn(input_size, device);
  at::Tensor input = torch::tensor(
      {{{{-0.6171, 1.9573}, {-0.9776, -0.4388}},
        {{0.0830 ,0.3295}, {1.1376 ,1.4564}}},
       {{{-0.7016, 1.2533}, {-0.2551, 0.3261}},
        {{1.3227, -0.1871}, {-1.0956, -1.0137}}}},device);
  //at::Tensor weight = torch::randn(weight_size, device);
  at::Tensor weight = torch::tensor(
      {{{{{-0.4548, - 0.1107}, {0.0223, - 1.2321}},
         {{1.1203, 1.7528}, {1.7692, - 0.9271}}},
        {{{1.3980, - 0.7515}, {1.2593, 1.0403}},
         {{0.3530, - 0.7153}, {1.7454 ,0.9942}}}},
       {{{{-0.7186, - 0.2611}, {0.6274 ,0.6565}},
         {{-1.4249, - 0.6673}, {-0.7796, - 1.1311}}},
        {{{0.3173, - 0.1052}, {-0.3449, 0.4073}},
         {{-0.7267, 0.0879}, {2.5294, 0.0152}}}},
       {{{{-1.3843, 1.6123}, {0.2952 ,- 0.7957}},
         {{-0.1807, 0.3354}, {0.8913 ,- 0.3995}}},
        {{{1.0847 ,0.5221}, {-0.3060 ,- 0.7522}},
         {{0.0249 ,- 0.5351}, {1.1408, 0.6337}}}}},device);
  //at::Tensor bias_opt = torch::randn({3}, device);
  at::Tensor bias_opt = torch::tensor({-0.8054,-1.1183,0.1813},device);
  std::vector<int64_t> stride_size = {1, 1, 1};
  std::vector<int64_t> dilation_size = {1, 1, 1};
  std::vector<int64_t> pad_size = {1, 1, 1};
  at::Tensor output_ref = at::slow_conv_dilated3d(
      input,
      weight,
      kernel_size,
      bias_opt,
      stride_size,
      pad_size,
      dilation_size);
  std::cout<<"\noutput_ref\n"<<output_ref;
}