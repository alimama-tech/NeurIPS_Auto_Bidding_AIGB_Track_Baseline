import numpy as np


class OfflineEnv:
    """
    Simulate an advertising bidding environment.
    """

    def __init__(self, min_remaining_budget: float = 0.1):
        """
        Initialize the simulation environment.
        :param min_remaining_budget: The minimum remaining budget allowed for bidding advertiser.
        """
        self.min_remaining_budget = min_remaining_budget

    def simulate_ad_bidding(self, pValues: np.ndarray,pValueSigmas: np.ndarray, bids: np.ndarray, leastWinningCosts: np.ndarray):
        """
        Simulate the advertising bidding process.

        :param pValues: Values of each pv .
        :param pValueSigmas: uncertainty of each pv .
        :param bids: Bids from the bidding advertiser.
        :param leastWinningCosts: Market prices for each pv.
        :return: Win values, costs spent, and winning status for each bid.

        """
        tick_status = bids >= leastWinningCosts
        tick_cost = leastWinningCosts * tick_status
        values = np.random.normal(loc=pValues, scale=pValueSigmas)
        values = values*tick_status
        tick_value = np.clip(values,0,1)
        tick_conversion = np.random.binomial(n=1, p=tick_value)

        return tick_value, tick_cost, tick_status,tick_conversion


def test():
    pv_values = np.array([10, 20, 30, 40, 50])
    pv_values_sigma = np.array([1, 2, 3, 4, 5])
    bids = np.array([15, 20, 35, 45, 55])
    market_prices = np.array([12, 22, 32, 42, 52])

    env = OfflineEnv()
    tick_value, tick_cost, tick_status,tick_conversion = env.simulate_ad_bidding(pv_values, bids, market_prices)

    print(f"Tick Value: {tick_value}")
    print(f"Tick Cost: {tick_cost}")
    print(f"Tick Status: {tick_status}")


if __name__ == '__main__':
    test()
