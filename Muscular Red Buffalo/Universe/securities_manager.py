from QuantConnect.Algorithm import QCAlgorithm
from datetime import timedelta
from AlgorithmImports import *


class SecuritiesManager:

    def __init__(self, algorithm: QCAlgorithm) -> None:

        """
        Args:
            algorithm: The algorithm instance.

        Note:
            Functionality as of 2023-09-18:
                * Able to process futures
                * Able to process Equity
        """

        self.algo = algorithm
    
    # Check if we have any futures in our universe
    def check_universe(self, data: object) -> str:
        chain = data.FuturesChains
        if chain:
            return self.check_for_rotation(data)
        else:
            return self.algo.tradable_instrument.Symbol

    # Futures management
    def check_for_rotation(self, data: object) -> str:
        to_symbol = self.get_current_futures_symbol(data)
        self.rotate_open_orders(data, to_symbol)
        self.rotate_holdings(to_symbol)
        return to_symbol

    def get_current_futures_symbol(self, data: object) -> str:
        for chain in data.FutureChains:

            contracts = list(filter(
                            lambda x: x.Expiry 
                            > self.algo.Time + timedelta(days=10), 
                            chain.Value
                            ))
            
            if len(contracts) == 0: continue
            front = sorted(contracts, key = lambda x: x.Expiry, reverse=False)[0]
            symbol = front.Symbol

            return symbol

    def get_open_orders(self) -> [Order]:
        open_orders = self.algo.Transactions.GetOpenOrders()
        return open_orders

    def rotate_open_orders(self, data: object, to_symbol):
        open_orders = self.get_open_orders()

        for order in open_orders:
            if order.Symbol != to_symbol:
                self.rotate_order(data, order, to_symbol)

    def rotate_order(self, data, order, to_symbol):
        current_price = order.Price

        self.algo.Transactions.CancelOrder(order.Id, orderTag=order.Tag)

        from_symbol_last_price = data[order.Symbol].Close
        to_symbol_last_price = data[to_symbol].Close
    
        price_difference = to_symbol_last_price - from_symbol_last_price
        new_price = self.round_to_minimum_tick_size(current_price + price_difference)

        if order.Type == 2: 
            
            # Stop Market Order
            self.algo.StopMarketOrder(
                symbol = to_symbol, 
                quantity = order.Quantity, 
                stopPrice = new_price, 
                tag = order.Tag
                )

        if order.Type == 1: 
            
            # Limit Order
            self.algo.LimitOrder(
                symbol = to_symbol, 
                quantity = order.Quantity, 
                limitPrice = new_price, 
                tag = order.Tag
                )
    
    def rotate_holdings(self, to_symbol) -> None:

        for alpha_model in self.algo.alpha_models:
            quantity = alpha_model.quantity
            symbol = alpha_model.symbol
            
            if symbol != to_symbol and quantity != 0:
            
                # Liquidate current position
                self.algo.MarketOrder(
                    symbol=symbol, 
                    quantity=-quantity, 
                    tag=f"{alpha_model.model_name} - Rotation"
                    )

                # Update to the next contract
                self.algo.MarketOrder(
                    symbol=to_symbol, 
                    quantity=quantity, 
                    tag=f"{alpha_model.model_name} - Rotation"
                    )
        return

    def round_to_minimum_tick_size(self, price: float) -> float:
        properties = self.algo.Securities[self.algo.current_symbol].SymbolProperties
        tick_size = properties.MinimumPriceVariation
        return tick_size * round(price / tick_size)

        
