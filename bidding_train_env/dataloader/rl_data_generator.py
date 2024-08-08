import os
import pandas as pd
import warnings
import glob

warnings.filterwarnings('ignore')


class RlDataGenerator:
    """
    RL Data Generator for RL models.
    Reads raw data and constructs training data suitable for reinforcement learning.
    """

    def __init__(self, file_folder_path="./data/traffic"):

        self.file_folder_path = file_folder_path
        self.training_data_path = self.file_folder_path + "/" + "training_data_rlData_folder"

    def batch_generate_rl_data(self):
        os.makedirs(self.training_data_path, exist_ok=True)
        csv_files = glob.glob(os.path.join(self.file_folder_path, '*.csv'))
        print(csv_files)
        training_data_list = []
        for csv_path in csv_files:
            print("开始处理文件：", csv_path)
            df = pd.read_csv(csv_path)
            df_processed = self._generate_rl_data(df)
            csv_filename = os.path.basename(csv_path)
            trainData_filename = csv_filename.replace('.csv', '-rlData.csv')
            trainData_path = os.path.join(self.training_data_path, trainData_filename)
            df_processed.to_csv(trainData_path, index=False)
            training_data_list.append(df_processed)
            del df, df_processed
            print("处理文件成功：", csv_path)
        combined_dataframe = pd.concat(training_data_list, axis=0, ignore_index=True)
        combined_dataframe_path = os.path.join(self.training_data_path, "training_data_all-rlData.csv")
        combined_dataframe.to_csv(combined_dataframe_path, index=False)
        print("整合多天训练数据成功；保存至:", combined_dataframe_path)

    def _generate_rl_data(self, df):
        """
        Construct a DataFrame in reinforcement learning format based on the raw data.

        Args:
            df (pd.DataFrame): The raw data DataFrame.

        Returns:
            pd.DataFrame: The constructed training data in reinforcement learning format.
        """

        training_data_rows = []

        for (
                deliveryPeriodIndex, advertiserNumber, advertiserCategoryIndex, budget,
                CPAConstraint), group in df.groupby(
            ['deliveryPeriodIndex', 'advertiserNumber', 'advertiserCategoryIndex', 'budget', 'CPAConstraint']):

            group = group.sort_values('timeStepIndex')

            group['timeStepIndex_volume'] = group.groupby('timeStepIndex')['timeStepIndex'].transform('size')

            timeStepIndex_volume_sum = group.groupby('timeStepIndex')['timeStepIndex_volume'].first()

            historical_volume = timeStepIndex_volume_sum.cumsum().shift(1).fillna(0).astype(int)
            group['historical_volume'] = group['timeStepIndex'].map(historical_volume)

            last_3_timeStepIndexs_volume = timeStepIndex_volume_sum.rolling(window=3, min_periods=1).sum().shift(
                1).fillna(0).astype(int)
            group['last_3_timeStepIndexs_volume'] = group['timeStepIndex'].map(last_3_timeStepIndexs_volume)

            group_agg = group.groupby('timeStepIndex').agg({
                'bid': 'mean',
                'leastWinningCost': 'mean',
                'conversionAction': 'mean',
                'xi': 'mean',
                'pValue': 'mean',
                'timeStepIndex_volume': 'first'
            }).reset_index()

            for col in ['bid', 'leastWinningCost', 'conversionAction', 'xi', 'pValue']:
                group_agg[f'avg_{col}_all'] = group_agg[col].expanding().mean().shift(1)
                group_agg[f'avg_{col}_last_3'] = group_agg[col].rolling(window=3, min_periods=1).mean().shift(1)

            group = group.merge(group_agg, on='timeStepIndex', suffixes=('', '_agg'))
            # 计算 realCost 和 realConversion
            realAllCost = (group['isExposed'] * group['cost']).sum()
            realAllConversion = group['conversionAction'].sum()



            for timeStepIndex in group['timeStepIndex'].unique():
                current_timeStepIndex_data = group[group['timeStepIndex'] == timeStepIndex]

                timeStepIndexNum = 48
                current_timeStepIndex_data.fillna(0, inplace=True)
                budget = current_timeStepIndex_data['budget'].iloc[0]
                remainingBudget = current_timeStepIndex_data['remainingBudget'].iloc[0]
                timeleft = (timeStepIndexNum - timeStepIndex) / timeStepIndexNum
                bgtleft = remainingBudget / budget if budget > 0 else 0

                state_features = current_timeStepIndex_data.iloc[0].to_dict()

                state = (
                    timeleft, bgtleft,
                    state_features['avg_bid_all'],
                    state_features['avg_bid_last_3'],
                    state_features['avg_leastWinningCost_all'],
                    state_features['avg_pValue_all'],
                    state_features['avg_conversionAction_all'],
                    state_features['avg_xi_all'],
                    state_features['avg_leastWinningCost_last_3'],
                    state_features['avg_pValue_last_3'],
                    state_features['avg_conversionAction_last_3'],
                    state_features['avg_xi_last_3'],
                    state_features['pValue_agg'],
                    state_features['timeStepIndex_volume_agg'],
                    state_features['last_3_timeStepIndexs_volume'],
                    state_features['historical_volume']
                )

                total_bid = current_timeStepIndex_data['bid'].sum()
                total_value = current_timeStepIndex_data['pValue'].sum()
                action = total_bid / total_value if total_value > 0 else 0
                reward = current_timeStepIndex_data[current_timeStepIndex_data['isExposed'] == 1][
                    'conversionAction'].sum()
                reward_continuous = current_timeStepIndex_data[current_timeStepIndex_data['isExposed'] == 1][
                    'pValue'].sum()

                done = 1 if timeStepIndex == timeStepIndexNum - 1 or current_timeStepIndex_data['isEnd'].iloc[
                    0] == 1 else 0

                training_data_rows.append({
                    'deliveryPeriodIndex': deliveryPeriodIndex,
                    'advertiserNumber': advertiserNumber,
                    'advertiserCategoryIndex': advertiserCategoryIndex,
                    'budget': budget,
                    'CPAConstraint': CPAConstraint,
                    'realAllCost':realAllCost,
                    'realAllConversion': realAllConversion,
                    'timeStepIndex': timeStepIndex,
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'reward_continuous': reward_continuous,
                    'done': done
                })

        training_data = pd.DataFrame(training_data_rows)
        training_data = training_data.sort_values(by=['deliveryPeriodIndex', 'advertiserNumber', 'timeStepIndex'])

        training_data['next_state'] = training_data.groupby(['deliveryPeriodIndex', 'advertiserNumber'])['state'].shift(
            -1)
        training_data.loc[training_data['done'] == 1, 'next_state'] = None
        return training_data


def generate_rl_data():
    file_folder_path = "./data/traffic"
    data_loader = RlDataGenerator(file_folder_path=file_folder_path)
    data_loader.batch_generate_rl_data()


if __name__ == '__main__':
    generate_rl_data()
