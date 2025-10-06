batches = 10000
batch_size = 4096
learning_rate = 1e-3
gamma = 1.0  # No discounting

version = "0_3"

save_path = f'models/model_v{version}.pth'

device_name = "cpu"

log_freq = 2
save_freq = 50