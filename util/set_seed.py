import random
import os
import numpy as np
import torch


def seed_all(seed, device='cuda'):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        torch.mps.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
    