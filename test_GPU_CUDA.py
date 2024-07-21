import torch
if torch.cuda.is_available():
    print("GPU is available.")
else:
    print("GPU is not available!")
    num_gpus = torch.cuda.device_count()
    print("number of GPUs:", num_gpus)