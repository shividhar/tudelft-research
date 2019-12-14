import torch
print(torch._C._cuda_getDriverVersion())
print(torch.cuda.is_available())
torch.cuda.current_device()
