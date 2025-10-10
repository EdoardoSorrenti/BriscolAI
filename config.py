import torch

batches = 10000
batch_size = 4096 * 2
learning_rate = 2.5e-4
gamma = 1.0  # No discounting

save_results = True

version = "0_3"

save_path = f'models/270k_{version}.pth'


device = torch.device("cuda")
dtype = torch.bfloat16

log_freq = 250
save_freq = 1000

optimize_points = True  # If True, optimize points directly instead of win/loss