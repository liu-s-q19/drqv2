import torch

print("Torch version:", torch.__version__)

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)

# 检查可用 GPU 数量
gpu_count = torch.cuda.device_count()
print("Number of CUDA devices:", gpu_count)

if cuda_available and gpu_count > 0:
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # 测试张量在 GPU 上运算
    device = torch.device("cuda")
    a = torch.rand(3, 3).to(device)
    b = torch.rand(3, 3).to(device)
    c = a + b
    print("Tensor computation on GPU succeeded.")
    print("Result:\n", c)
else:
    print("CUDA is not available or no GPU detected. Running on CPU.")
    # 测试 CPU 运算
    a = torch.rand(3, 3)
    b = torch.rand(3, 3)
    c = a + b
    print("Result:\n", c)
