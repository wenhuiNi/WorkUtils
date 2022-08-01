#include <c10/util/Optional.h>
#include <torch/torch.h>
#include <tuple>
#include <iostream>
#include <vector>

void gather_cuda_kernel_test() {
  auto device = torch::kCUDA;
  std::vector<int64_t> size = {2,2};
  at::Tensor self = torch::randn(size,device);
  at::Tensor result = torch::randn(size,device);
  int64_t dim = 1;
  at::Tensor index = torch::tensor({{0,0},{1,0}},device);
  std::cout<<result<<'\n'<<self<<'\n';
  at::native::gather_cuda_kernel(result, self, dim, index);
  std::cout<<result<<'\n'<<self<<'\n';
}

int main(){
  gather_cuda_kernel_test();
}