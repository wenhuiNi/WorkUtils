#include <c10/util/Optional.h>
#include <torch/torch.h>
#include <tuple>
#include <iostream>
#include <vector>

void output_thnn_fused_lstm_cell_cuda(){
  auto device = torch::kCUDA;
  std::cout<<"\n______"<<__func__<<"____________\n";
  std::vector<int64_t> input_size = {2,2};
  std::vector<int64_t> cx_size = {2,2};
  std::vector<int64_t> bias_size = {2};

  at::Tensor input_gates = torch::tensor({{0.2227 , 0.8805},{1.2745 , 0.5710}});

  at::Tensor hidden_gates= torch::tensor({{1.9588 ,-2.2766},{1.0055  ,1.4349}});
  at::Tensor cx = at::randn({1,1});
  cx[0,0]=0.7804;
  at::Tensor input_bias = torch::tensor({0.6773,-1.7560});

  at::Tensor hidden_bias = torch::tensor({0.2444,0.5332});

  at::Tensor output_ref1=torch::tensor({{0.6676}});
  at::Tensor output_ref2=torch::tensor({{0.9902}});
  at::Tensor output_ref3=torch::tensor({{0.9570, 0.0679},{0.9793 ,0.8814}});

  at::Tensor input_device=input_gates.to(device);
  at::Tensor hidden_device=hidden_gates.to(device);
  at::Tensor cx_device=cx.to(device);
  at::Tensor input_bias_device=input_bias.to(device);
  at::Tensor hidden_bias_device= hidden_bias.to(device);

  std::tuple<at::Tensor, at::Tensor, at::Tensor> output= 
    at::_thnn_fused_lstm_cell(input_device,hidden_device,cx_device,input_bias_device,hidden_bias_device);
}

void output_thnn_fused_lstm_cell_backward(){
  auto device = torch::kCUDA;
  std::cout<<"\n______"<<__func__<<"____________\n";
  std::vector<int64_t> hy_size = {2,2};
  std::vector<int64_t> cy_size = {2,2};
  std::vector<int64_t> workspace_size = {4,4};

  at::Tensor grad_hy_opt = torch::tensor({{-0.8938, -1.7885},{0.5743, 0.6261}});
  at::Tensor grad_cy_opt = torch::tensor({{-0.4043 ,-1.2028},{1.2103  ,1.9045}});
  at::Tensor workspace = torch::tensor({{1.2366 ,0.7817 , 0.5665, -0.0782},
  {0.0002 , 1.4007  ,0.7640 , 0.2485},
  {-1.2833 , 0.7944 ,-0.4622 , 1.2357},
  {0.1524 ,-0.1197, -0.4770 ,-1.2804}});
  at::Tensor cx = torch::tensor({{-0.7809 ,-0.1163},{0.4453 ,0.3882}});
  at::Tensor cy = torch::tensor({{1.1261, -0.2067},{0.0910  ,0.2323}});

  /*
  at::Tensor grad_hy_opt = at::randn(hy_size);
  at::Tensor grad_cy_opt = at::randn(cy_size);
  at::Tensor workspace = at::randn(workspace_size);
  at::Tensor cx = at::randn({2,2});
  at::Tensor cy = at::randn({2,2});
  */

  at::Tensor grad_hy_device = grad_hy_opt.to(device);
  at::Tensor grad_cy_device = grad_cy_opt.to(device);
  at::Tensor workspace_device = workspace.to(device);
  at::Tensor cx_device = cx.to(device);
  at::Tensor cy_device = cy.to(device);

  // std::cout<<grad_hy_opt<<'\n';
  // std::cout<<grad_cy_opt<<'\n';
  // std::cout<<workspace<<'\n';
  // std::cout<<cx<<'\n';
  // std::cout<<cy<<'\n';

  at::Tensor output1 = torch::tensor({{2.9092e-05, -3.8932e-01,  1.2265e-01, -1.5972e-02},
  {-7.9084e-01,  1.2250e+00, -1.3049e-01,  6.8076e-0},
  {-4.1917e-01, -2.2378e-02, -2.8247e-01, -1.2943e-01},
  {-1.1766e+00,  8.9620e-01, -3.6706e-02, -4.1712e-01}});
  at::Tensor output2 = torch::tensor({{2.9092e-05, -3.8932e-01,  1.2265e-01, -1.5972e-02},
  {-7.9084e-01,  1.2250e+00, -1.3049e-01,  6.8076e-02},
  {-4.1917e-01, -2.2378e-02, -2.8247e-01, -1.2943e-01},
  {-1.1766e+00,  8.9620e-01, -3.6706e-02, -4.1712e-01}});
  at::Tensor output3 = torch::tensor({{-0.3623 , 0.1274},{-0.4338 , 1.4143}});
  at::Tensor output4 = torch::tensor({-2.3865,1.7095,-0.3270,-0.4944});
  at::Tensor output5 = torch::tensor({-2.3865,1.7095,-0.3270,-0.4944});


  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> output=
  at::_thnn_fused_lstm_cell_backward(grad_hy_device,grad_cy_device,cx_device,cy_device,workspace_device,true);

  //std::cout<<std::get<0>(output)<<'\n'<<std::get<1>(output)<<'\n'<<std::get<2>(output)<<'\n'<<std::get<3>(output)<<'\n'<<std::get<4>(output)<<'\n';
  
}

void  output_thnn_fused_gru_cell_cuda(){
  auto device = torch::kCUDA;
  std::cout<<"\n______"<<__func__<<"____________\n";
  at::Tensor input_gates = torch::tensor({{0.2227 , 0.8805},{1.2745 , 0.5710}});

  at::Tensor hidden_gates= torch::tensor({{1.9588 ,-2.2766},{1.0055  ,1.4349}});
  at::Tensor cx = at::randn({1,1});
  cx[0,0]=0.7804;
  at::Tensor input_bias = torch::tensor({0.6773,-1.7560});

  at::Tensor hidden_bias = torch::tensor({0.2444,0.5332});

  at::Tensor input_device=input_gates.to(device);
  at::Tensor hidden_device=hidden_gates.to(device);
  at::Tensor cx_device=cx.to(device);
  at::Tensor input_bias_device=input_bias.to(device);
  at::Tensor hidden_bias_device= hidden_bias.to(device);

  std::tuple<at::Tensor,  at::Tensor> output= 
    at::_thnn_fused_gru_cell(input_device,hidden_device,cx_device,input_bias_device,hidden_bias_device);

  std::cout<<std::get<0>(output)<<'\n'<<std::get<1>(output)<<'\n';

  at::Tensor output1 = at::randn({1,1});
  output1[0,0]=0.9778;
  at::Tensor output2 = torch::tensor({{0.9570, 0.0679,  0.9922 , 0.7804  ,1.0965}});
  //std::cout<<output1<<output2;

}

void  output_thnn_fused_gru_cell_backward_cuda(){
  auto device = torch::kCUDA;
  std::cout<<"\n______"<<__func__<<"____________\n";
  at::Tensor grad_hy_opt = torch::tensor({{-0.8938, -1.7885},{0.5743, 0.6261}});
  at::Tensor workspace = torch::tensor(
    {{1.0855 ,-1.5694 , 0.0744, -0.8877, -1.2372 , 0.8160 ,-0.0833, -0.5543,  1.8053, -2.5755},
    {-0.9299, -1.3975, -0.5028,  0.8479, -0.4355, -0.0010 , 0.7194,  0.7309, -0.9081, -1.8823}});

  at::Tensor grad_hy_device = grad_hy_opt.to(device);

  at::Tensor workspace_device = workspace.to(device);

  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> output=
  at::_thnn_fused_gru_cell_backward(grad_hy_device,workspace_device,true);
  std::cout<<std::get<0>(output)<<'\n'<<std::get<1>(output)<<'\n'<<std::get<2>(output)<<'\n'<<std::get<3>(output)<<'\n'<<std::get<4>(output)<<'\n';
  
  at::Tensor output1 = torch::tensor({{-0.0736, -11.7161, -0.0710, -4.1068,  0.4390, -1.1281},
  {1.1398 , 0.6006 ,-0.5012 , 0.0591 , 0.6994 , 0.0952}});
  at::Tensor output2 = torch::tensor({{-0.0736 ,-11.7161, -0.0710, -4.1068 , 0.4766 , 1.7705},
  {1.1398,  0.6006, -0.5012,  0.0591 ,-0.6503, -0.1331}});
  at::Tensor output3 = torch::tensor({{-0.0665 , 1.5877},{-0.2888 ,0.5309}});
  at::Tensor output4 = torch::tensor({1.0662,-11.1155,-0.5722,-4.0477,1.1384,-1.0329});
  at::Tensor output5 = torch::tensor({1.0662,-11.1155,-0.5722,-4.0477,-0.1738, 1.6374});

}

int main(){
  output_thnn_fused_lstm_cell_cuda();
  output_thnn_fused_lstm_cell_backward();
  output_thnn_fused_gru_cell_cuda();
  output_thnn_fused_gru_cell_backward_cuda();
}