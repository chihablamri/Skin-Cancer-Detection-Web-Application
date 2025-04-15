import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA Device Count:", torch.cuda.device_count())
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
    print("CUDA Device Capability:", torch.cuda.get_device_capability(0))
    
    # Test GPU computation
    print("\nRunning test computation on GPU...")
    x = torch.rand(5, 3).cuda()
    y = torch.rand(3, 5).cuda()
    z = torch.matmul(x, y)
    print("Test computation result shape:", z.shape)
    print("Test computation result:\n", z)
    print("\nGPU test successful!")
else:
    print("No CUDA GPU detected by PyTorch.")
    print("Please check your NVIDIA drivers and CUDA installation.") 