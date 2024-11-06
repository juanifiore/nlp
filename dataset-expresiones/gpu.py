import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU: ", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Usando cpu")
