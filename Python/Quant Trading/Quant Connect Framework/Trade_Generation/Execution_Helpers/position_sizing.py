from AlgorithmImports import *
from Helpers.enums import Direction


class PositionSizing():

    def __init__(self, algorithm) -> None:

        """
        Args:
            algorithm: The main algorithm instance

        Note: QuantityHelper is responsible for the position size.
        select one of the defined functions. Note- different Alpha Models
        can use different quantity functions at the same algorithm

        """

        self.algo = algorithm

    def margin_check(func):
        """Decorator to ensure position size doesn't exceed available margin."""
        
        def wrapper(self, *args, **kwargs):
            
            # 1. Call the original function
            position_size = func(self, *args, **kwargs)
            
            # 2. Retrieve price, multiplier and margin
            symbol = self.algo.current_symbol
            price = self.algo.Securities[symbol].Close
            multiplier = self.algo.Securities[symbol].SymbolProperties.ContractMultiplier
            max_trade_value = self.algo.Portfolio.MarginRemaining

            # 3. Check and adjust position size based on available margin
            if abs(position_size * price * multiplier) > max_trade_value:
                adjusted_position_size = int(max_trade_value / (price * multiplier))
                return adjusted_position_size if position_size > 0 else -adjusted_position_size
            
            return position_size
            
        return wrapper

    @margin_check
    def equal_qty_per_model(self, direction: Direction) -> int:

        """
        Note:
            Function calculates position size equally among all Alpha Models
            so that if every model has a position- it will not exceed the
            total portfolio value
        """

        symbol = self.algo.current_symbol
        price = self.algo.Securities[symbol].Close
        multiplier = self.algo.Securities[symbol].SymbolProperties.ContractMultiplier
        alpha_models_count = len(self.algo.alpha_models)
        exposure_per_model = 1 / alpha_models_count
        notional_amount = exposure_per_model * self.algo.Portfolio.TotalPortfolioValue
        self.algo.Log(f"{self.algo.Time} - remaining margin:  {self.algo.Portfolio.MarginRemaining}")
        if direction == Direction.Long:
            return int((notional_amount) / (price * multiplier))
        else:
            return -int((notional_amount) / (price * multiplier))
    
    @margin_check
    def full_portfolio_per_model(self) -> int:

        """
        Note:
            Function calculates position size to be equal to 100% of the total
            portfolio value. Meaning if 2 Alpha Models are in the algorithm and
            both have a position, the total value will be twice that of portfolio
            - Will need leverage to employ
        """

        symbol = self.algo.current_symbol
        price = self.algo.Securities[symbol].Close
        multiplier = self.algo.Securities[symbol].SymbolProperties.ContractMultiplier
        alpha_models_count = len(self.algo.alpha_models)
        exposure_per_model = 1
        notional_amount = exposure_per_model * self.algo.Portfolio.TotalPortfolioValue
        return int((notional_amount) / (price * multiplier))
    
    @margin_check
    def percentage_from_portfolio(self, percentage: str("5%: 0.05")) -> int:

        """
        Note:
            Function calculates position size depending on specified percentage. 
            Note, however, that if we have 5 Alpha Models who each can have up to
            1 position at a time and this position sizing algorithm is set to 
            work on 10% from portfolio value- we will never employ full capital
        """

        symbol = self.algo.current_symbol
        price = self.algo.Securities[symbol].Close
        multiplier = self.algo.Securities[symbol].SymbolProperties.ContractMultiplier
        notional_amount = self.algo.Portfolio.TotalPortfolioValue * percentage
        return int(notional_amount / (price * multiplier))
    
    @margin_check
    def percentage_from_given_value(self, percentage: str("5%: 0.05"), value: float) -> int:

        """
        Note:
            Function calculates position size depending on specified percentage from
            specified value
        """

        symbol = self.algo.current_symbol
        price = self.algo.Securities[symbol].Close
        multiplier = self.algo.Securities[symbol].SymbolProperties.ContractMultiplier
        notional_amount = value * percentage
        return int(notional_amount / (price * multiplier))
    
    @margin_check
    def specified_contract_amount(self, amount) -> int:

        """
        Note:
            Function simply returns the amount we passed
        """

        return amount
    
    @margin_check
    def kelly_criterion(self) -> int:
        return