#include <torch/torch.h>
#include <tuple>
#include <iostream>
#include <vector>
int main(){
auto device = torch::kCUDA;
  std::vector<int64_t> size = {5,5};
  int64_t dim = -2;
  bool keepdim = false;
  at::Tensor self = torch::randn(size,device);
  at::Tensor values = torch::randn({5},self.options());
  at::Tensor indices = torch::tensor({0,0,0,0,0},device);
  std::tuple<at::Tensor, at::Tensor> output_ref = at::median_out(values,indices,self,dim,keepdim);
std::cout<<std::get<0>(output_ref)<<'\n'<<std::get<1>(output_ref)<<'\n';

}
