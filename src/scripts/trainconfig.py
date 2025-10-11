import torch

# Logging configuration
log_level = "INFO"
log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
log_datefmt = "%Y-%m-%d %H:%M:%S"
log_to_file = False
log_file_path = "logs/selfplay.log"

batches = None
batch_size = 4096 * 2
learning_rate = 2.5e-4
gamma = 1.0  # No discounting

save_results = True

version = "float32"

save_path = f'training_models/270k_{version}.pth'


device = torch.device("cuda")
dtype = torch.float32
tensor_f32 = True

log_freq = 1000
save_freq = log_freq * 10
eval_freq = log_freq * 10
eval_batches = 10  # Number of batches to evaluate candidate model

optimize_points = True  # If True, optimize points directly instead of win/loss