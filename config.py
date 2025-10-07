batches = 10000
batch_size = 4096 * 2
learning_rate = 8e-4
gamma = 1.0  # No discounting

save_results = True

version = "0_3"

save_path = f'models/model_v{version}.pth'

device_name = "cuda"

log_freq = 10
save_freq = 50