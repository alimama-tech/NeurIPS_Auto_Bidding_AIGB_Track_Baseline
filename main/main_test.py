import numpy as np
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run.run_evaluate import run_test

torch.manual_seed(1)
np.random.seed(1)

if __name__ == "__main__":
    run_test()
