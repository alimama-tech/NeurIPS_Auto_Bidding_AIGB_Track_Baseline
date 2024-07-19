import numpy as np
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run.run_decision_transformer import run_dt


torch.manual_seed(1)
np.random.seed(1)


if __name__ == "__main__":
    """程序主入口，运行BC算法"""
    run_dt()
