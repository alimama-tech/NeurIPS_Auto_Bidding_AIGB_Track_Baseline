import pandas as pd
import os
import pickle
import numpy as np


def normalize_state(training_data, state_dim, normalize_indices):
    """
    Normalize features for reinforcement learning.
    Args:
        training_data: A DataFrame containing the training data.
        state_dim: The total dimension of the features.
        normalize_indices: A list of indices of the features to be normalized.

    Returns:
        A dictionary containing the normalization statistics.
    """
    state_columns = [f'state{i}' for i in range(state_dim)]
    next_state_columns = [f'next_state{i}' for i in range(state_dim)]

    for i, (state_col, next_state_col) in enumerate(zip(state_columns, next_state_columns)):
        training_data[state_col] = training_data['state'].apply(
            lambda x: x[i] if x is not None and not np.isnan(x).any() else 0.0)
        training_data[next_state_col] = training_data['next_state'].apply(
            lambda x: x[i] if x is not None and not np.isnan(x).any() else 0.0)

    stats = {
        i: {
            'min': training_data[state_columns[i]].min(),
            'max': training_data[state_columns[i]].max(),
            'mean': training_data[state_columns[i]].mean(),
            'std': training_data[state_columns[i]].std()
        }
        for i in normalize_indices
    }

    for state_col, next_state_col in zip(state_columns, next_state_columns):
        if int(state_col.replace('state', '')) in normalize_indices:
            min_val = stats[int(state_col.replace('state', ''))]['min']
            max_val = stats[int(state_col.replace('state', ''))]['max']
            training_data[f'normalize_{state_col}'] = (
                                                              training_data[state_col] - min_val) / (
                                                              max_val - min_val + 0.01)
            training_data[f'normalize_{next_state_col}'] = (
                                                                   training_data[next_state_col] - min_val) / (
                                                                   max_val - min_val + 0.01)
            # 0.01 error too large?
        else:
            training_data[f'normalize_{state_col}'] = training_data[state_col]
            training_data[f'normalize_{next_state_col}'] = training_data[next_state_col]

    training_data['normalize_state'] = training_data.apply(
        lambda row: tuple(row[f'normalize_{state_col}'] for state_col in state_columns), axis=1)
    training_data['normalize_nextstate'] = training_data.apply(
        lambda row: tuple(row[f'normalize_{next_state_col}'] for next_state_col in next_state_columns), axis=1)

    return stats


def normalize_reward(training_data, reward_type):
    """
    Normalize rewards for reinforcement learning.

    Args:
        training_data: A DataFrame containing the training data.
        reward_type: reward:sparse reward   reward_continuous: continuous reward

    Returns:
        A Series of normalized rewards.
    """
    reward_range = training_data[reward_type].max() - training_data[reward_type].min() + 0.00000001
    training_data["normalize_reward"] = (
                                                training_data[reward_type] - training_data[
                                            reward_type].min()) / reward_range
    return training_data["normalize_reward"]


def save_normalize_dict(normalize_dict, save_dir):
    """
    Save the normalization dictionary to a Pickle file.

    Args:
        normalize_dict: The dictionary containing normalization statistics.
        save_dir: The directory to save the normalization dictionary.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'normalize_dict.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(normalize_dict, file)


if __name__ == '__main__':
    test_data = {
        'state': [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
        'next_state': [(2, 3, 4), (5, 6, 7), (8, 9, 10)],
        'reward': [10, 20, 30]
    }
    training_data = pd.DataFrame(test_data)
    state_dim = 3
    normalize_indices = [0, 2]
    stats = normalize_state(training_data, state_dim, normalize_indices)
    normalize_reward(training_data)
    print(training_data)
    print(stats)
