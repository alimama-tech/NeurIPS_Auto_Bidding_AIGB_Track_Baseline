from torch.utils.data import Dataset
import pandas as pd
import ast
import numpy as np
import torch



class aigb_dataset(Dataset):
    def __init__(self, step_len, **kwargs) -> None:
        super().__init__()
        states, actions, rewards, terminals = load_local_data_nips(
            train_data_path="data/trajectory/trajectory_data.csv")
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        self.step_len = step_len
        self.num_of_states = states.shape[1]
        self.num_of_actions = actions.shape[1]

        # 分割序列
        # 每个序列的开头
        self.candidate_pos = (self.terminals == 0).nonzero()[0]
        self.candidate_pos += 1
        self.candidate_pos = [0] + self.candidate_pos.tolist()[:-1]
        # 后面再加上序列的结尾
        self.candidate_pos = self.candidate_pos + [self.states.shape[0]]

    def __len__(self):
        return len(self.candidate_pos) - 1

    def __getitem__(self, index):
        # 获取序列
        state = torch.tensor(self.states[self.candidate_pos[index]:self.candidate_pos[index + 1], :],
                             dtype=torch.float32)
        action = torch.tensor(self.actions[self.candidate_pos[index]:self.candidate_pos[index + 1], :],
                              dtype=torch.float32)
        reward = torch.tensor(self.rewards[self.candidate_pos[index]:self.candidate_pos[index + 1], :],
                              dtype=torch.float32)
        action = action - 1
        # 当前序列的长度
        len_state = len(state)
        # 进行padding
        state = torch.nn.functional.pad(state, (0, 0, 0, self.step_len - len(state)), "constant", 0)
        action = torch.nn.functional.pad(action, (0, 0, 0, self.step_len - len(action)), "constant", 0)
        # 计算returns
        returns = reward.sum().sigmoid()
        returns = torch.clamp(returns, max=1.0).reshape(1)
        # 计算masks
        masks = torch.zeros(self.step_len)
        masks[:len_state] = 1
        masks = masks.bool()
        # 返回
        return state, action, returns, masks


# 加载本地数据
def load_local_data(data_version):
    states = pd.read_csv("simulation_platform/data/offline_trajectory/" + data_version + "/states.csv").values[:,
             0::]
    actions = pd.read_csv("simulation_platform/data/offline_trajectory/" + data_version + "/actions.csv").values[:,
              0::]
    rewards = pd.read_csv("simulation_platform/data/offline_trajectory/" + data_version + "/rewards.csv").values[:,
              0::]
    terminals = pd.read_csv("simulation_platform/data/offline_trajectory/" + data_version + "/terminal.csv").values[
                :,
                0::]
    return states, actions, rewards, terminals


def load_local_data_nips(train_data_path="data/traffic/training_data_rlData_folder/training_data_all-rlData.csv"):
    training_data = pd.read_csv(train_data_path)

    def safe_literal_eval(val):
        if pd.isna(val):
            return val  # 如果是NaN，返回NaN
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            print(ValueError)
            return val  # 如果解析出错，返回原值

    # 使用apply方法应用上述函数
    training_data["state"] = training_data["state"].apply(safe_literal_eval)
    training_data["terminal"] = training_data["timeStepIndex"] != 47
    training_data["terminal"] = training_data["terminal"].astype(int)
    states = np.array(training_data['state'].tolist())
    actions = training_data["action"].to_numpy().reshape(-1, 1)
    rewards = training_data["reward"].to_numpy().reshape(-1, 1)
    terminals = training_data["terminal"].to_numpy().reshape(-1, 1)
    return states, actions, rewards, terminals
