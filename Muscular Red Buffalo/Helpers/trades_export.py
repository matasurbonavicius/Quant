from AlgorithmImports import *


class TradesExport:
    
    def __init__(self, algorithm) -> None:

        """
        Args:
            algorithm: The main algorithm instance
        """

        self.algo = algorithm
        self.trade_number = 1
        self.log_journal = {}

    def put_buy_trade(self) -> None:
        if self.trade_number not in self.log_journal:
            self.log_journal[self.trade_number] = {}
        self.log_journal[self.trade_number]["Buy time"] = self.algo.Time
    
    def put_sell_trade(self) -> None:
        if self.trade_number not in self.log_journal:
            self.log_journal[self.trade_number] = {}
        self.log_journal[self.trade_number]["Sell time"] = self.algo.Time
        self.trade_number += 1

    def print_all_trades(self) -> None:
        for key, value in self.log_journal.items():
            buy_time = value.get("Buy time", "N/A")
            sell_time = value.get("Sell time", "N/A")
            self.algo.Log(
                f"Trade Number, {key}, Buy date, {buy_time}, Sell date, {sell_time}"
                )
