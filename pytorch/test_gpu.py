import torch
print(torch._C._cuda_getDriverVersion())
print(torch.cuda.is_available())
print(torch.cuda.nccl.version())
torch.cuda.current_device()
