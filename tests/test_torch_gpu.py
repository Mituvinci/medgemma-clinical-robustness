import torch
print(f"Is CUDA available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")


import torch
from accelerate import Accelerator

# Check if Accelerator sees both A40s
accelerator = Accelerator()
print(f"Number of GPUs detected: {accelerator.num_processes}")
print(f"Distributed type: {accelerator.distributed_type}")