# import time
#
import torch

#
# #a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device='cuda')
#
# # b = torch.FloatTensor(2, 2)
# #
# # c = torch.FloatTensor([2, 5, 3, 4])
#
#
# # # d = torch.Tensor(2, 5)
# # e = torch.Tensor(2, 5)
# # f = torch.Tensor(2, 5)
#
# # print(a)
# # print(b)
# # print(c)
# # # print(d)
# # print(e)
# # print(f)
#
# # a = torch.eye(2, 2)
# # c = torch.eye(4, 3)
# # b = torch.eye(5, 5)
# # print(a)
# # print(b)
# # print(c)
# # a = torch.randn(2, 2)
# # b = torch.randperm(4)
#
# # a = torch.randn(2,3)
# # b = torch.randn(3,2)
# # c = torch.mm(a,b)
# # print(a)
# # print(b)
# # print(c)
#
# # a = torch.cuda.FloatTensor(4, 3)
#
# # a = torch.IntTensor([2, 3, 4, 5, 6, 7, 8, 9, 10])
# # b = torch.rand(2, 4)
# # print(a)
# # print(b)
#
# a = torch.randn(2, 3)
# print(a)
# b = torch.randn(3)
# print(b)
# # c = torch.mm(a, b.T)
# # print(c)
# c = torch.mv(a, b)
# print(c)
#
# # b = torch.abs(a)
# # print(b)
# # c = torch.add(a, b)
# # print(c)
# # e = torch.add(c, 1)
# # print(e)
# #
# # d = torch.clamp(e, 1, 2)
# # print(d)
# # c = torch.div(a, b)
# # print(c)
# # d = torch.pow(a, 2)
# # print(d)
# from torch import nn
#
#
# class MLP(nn.Module):
#     def __init__(self, in_dim, hid_dim1, hid_dim2, out_dim):
#         super(MLP, self).__init__()
#         self.layer = nn.Sequential(
#             nn.Linear(in_dim, hid_dim1),
#             nn.ReLU(),
#             nn.Linear(hid_dim1, hid_dim2),
#             nn.ReLU(),
#             nn.Linear(hid_dim2, out_dim),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         x = self.layer(x)
#         return x
#
#
#
# x = torch.eye(4)
# print(x)
#
# y = torch.diag(x)
# print(y)
#
# n = x.size()
# print(n)
# m = x.shape
# print(m)
# # o = x.view(-1, 8)
# # print(o)
#
# a = torch.randn(3, 4)
# print(a)
# b = torch.randn(1, 1, 1).transpose(1, 2)
# print(b)
# print(b.dtype)


# 测试GPU环境是否可使用
# print(torch.__version__)  # pytorch版本
# print(torch.version.cuda)  # cuda版本
# print(torch.cuda.is_available())  # 查看cuda是否可用

# 使用GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建张量 a 和 b，并将它们移动到相同设备上
a = torch.tensor([1, 2, 3]).to(device)
b = torch.tensor([4, 5, 6]).to(device)

# 查看对象在哪个设备上运行
print("a is on device:", a.device)
print("b is on device:", b.device)

# # 将对象的环境设置为device环境
# A = torch.tensor([1, 2, 3])
# A = A.to(device)
# print(A.device)
#
# # 将对象环境设置为CPU
# A = A.cpu()
# print(A.device)

# 若一个没有环境的对象与另外一个有环境对象进行操作,则需要将它们放在同一设备上
c = a + b
print(c.device)

# cuda环境下tensor不能直接转化为numpy类型,必须要先转化到cpu环境中
a_cpu = a.cpu().numpy()
print(a_cpu)


# 创建CUDA型的tensor
cuda_tensor = torch.tensor([1, 2], device=device)
print(cuda_tensor.device)

