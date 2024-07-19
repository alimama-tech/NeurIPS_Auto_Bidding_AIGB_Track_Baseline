import os
import numpy as np
import pandas as pd
import pickle
import warnings

warnings.filterwarnings('ignore')


class TestDataLoader:
    """
    Offline evaluation data loader.
    """

    def __init__(self, file_path="./data/log.csv"):
        """
        Initialize the data loader.
        Args:
            file_path (str): The path to the training data file.

        """
        self.file_path = file_path
        self.raw_data_path = os.path.join(os.path.dirname(file_path), "raw_data.pickle")
        self.raw_data = self._get_raw_data()
        self.keys, self.test_dict = self._get_test_data_dict()

    def _get_raw_data(self):
        """
        Read raw data from a pickle file.

        Returns:
            pd.DataFrame: The raw data as a DataFrame.
        """
        if os.path.exists(self.raw_data_path):
            with open(self.raw_data_path, 'rb') as file:
                return pickle.load(file)
        else:
            tem = pd.read_csv(self.file_path)
            with open(self.raw_data_path, 'wb') as file:
                pickle.dump(tem, file)
            return tem

    def _get_test_data_dict(self):
        """
        Group and sort the raw data by deliveryPeriodIndex and advertiserNumber.

        Returns:
            list: A list of group keys.
            dict: A dictionary with grouped data.

        """
        grouped_data = self.raw_data.sort_values('timeStepIndex').groupby(['deliveryPeriodIndex', 'advertiserNumber'])
        data_dict = {key: group for key, group in grouped_data}
        return list(data_dict.keys()), data_dict

    def mock_data(self, key):
        """
        Get training data based on deliveryPeriodIndex and advertiserNumber, and construct the test data.
        """
        data = self.test_dict[key]
        pValues = data.groupby('timeStepIndex')['pValue'].apply(list).apply(np.array).tolist()
        pValueSigmas = data.groupby('timeStepIndex')['pValueSigma'].apply(list).apply(np.array).tolist()
        leastWinningCosts = data.groupby('timeStepIndex')['leastWinningCost'].apply(list).apply(np.array).tolist()
        num_timeStepIndex = len(pValues)
        return num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts


if __name__ == '__main__':
    pass
