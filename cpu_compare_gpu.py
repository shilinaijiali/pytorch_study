import torch
import time

# 定义设备
device_cpu = torch.device('cpu')
device_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建大张量
size = (10000, 10000)

# 在 CPU 上创建张量并进行操作
start_time = time.time()
a_cpu = torch.randn(size, device=device_cpu)
b_cpu = torch.randn(size, device=device_cpu)
c_cpu = a_cpu + b_cpu
end_time = time.time()
print(f"CPU operation took {end_time - start_time:.4f} seconds")

# 在 CUDA 上创建张量并进行操作
if torch.cuda.is_available():
    start_time = time.time()
    a_cuda = torch.randn(size, device=device_cuda)
    b_cuda = torch.randn(size, device=device_cuda)
    c_cuda = a_cuda + b_cuda
    torch.cuda.synchronize()  # 等待 CUDA 完成
    end_time = time.time()
    print(f"CUDA operation took {end_time - start_time:.4f} seconds")
else:
    print("CUDA not available")
