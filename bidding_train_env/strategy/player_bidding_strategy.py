import time
import numpy as np
import os
import psutil

from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy


class PlayerBiddingStrategy(BaseBiddingStrategy):
    """
    Simple Strategy example for bidding.
    """

    def __init__(self, budget=100, name="PlayerStrategy", cpa=40, category=1):
        """
        Initialize the bidding Strategy.
        parameters:
            @budget: the advertiser's budget for a delivery period.
            @cpa: the CPA constraint of the advertiser.
            @category: the index of advertiser's industry category.

        """
        super().__init__(budget, name, cpa, category)

    def reset(self):
        """
        Reset remaining budget to initial state.
        """
        self.remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        """
        Bids for all the opportunities in a delivery period

        parameters:
         @timeStepIndex: the index of the current decision time step.
         @pValues: the conversion action probability.
         @pValueSigmas: the prediction probability uncertainty.
         @historyPValueInfo: the history predicted value and uncertainty for each opportunity.
         @historyBids: the advertiser's history bids for each opportunity.
         @historyAuctionResult: the history auction results for each opportunity.
         @historyImpressionResult: the history impression result for each opportunity.
         @historyLeastWinningCosts: the history least wining costs for each opportunity.

        return:
            Return the bids for all the opportunities in the delivery period.
        """

        return self.cpa * pValues
