from AlgorithmImports import *
from Helpers.enums import Direction


class AbstractSLStrategy:
    
    """
    Base class for stop loss strategies.
    """

    def __init__(self, algo) -> None:
        self.algo = algo

    def apply(self, alpha_model: str, price: float, direction: Direction):
        raise NotImplementedError

    def place_order(self, alpha_model: str, sl_price: float):
        for model in self.algo.alpha_models:
            if alpha_model in model.model_name:
                current_model = model
                break

        self.algo.StopMarketOrder(
            symbol=self.algo.signal_instrument.Symbol,
            quantity=-current_model.quantity,
            stopPrice=sl_price,
            tag=f"{alpha_model} - Stop Loss"
        )


class StopLossManager:
    
    """
    Manages stop loss strategies for alpha models.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(StopLossManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, algorithm) -> None:
        self.algo = algorithm
        self.alpha_models_sl = {}

    def register_sl_strategy(self, alpha_model: str, strategy: AbstractSLStrategy):
        self.alpha_models_sl[alpha_model] = strategy
        self.algo.Debug(f"Registered SL Strategy for {alpha_model}")

    def use_sl(self, alpha_model, price, direction: Direction) -> None:
        strategy = self.alpha_models_sl.get(alpha_model)
        if strategy:
            strategy.apply(alpha_model, price, direction)


class CandleHighLowSLStrategy(AbstractSLStrategy):

    def __init__(self, algo, candle_high: float = None, candle_low: float = None) -> None:
        super().__init__(algo)
        self.candle_low = candle_low
        self.candle_high = candle_high

    def apply(self, alpha_model: str, price: float, direction: Direction):
        if direction == Direction.Long:
            self.place_order(alpha_model, self.candle_low)
        else:
            self.place_order(alpha_model, self.candle_high)


class NoSLStrategy(AbstractSLStrategy):

    """
    No SL Strategy in use
    """

    def __init__(self, algo) -> None:
        super().__init__(algo)
    
    def apply(self, alpha_model: str, price: float, direction: Direction):
        pass


class PercentageSLStrategy(AbstractSLStrategy):
    
    """
    Triggers stop loss based on a percentage of the current price.
    """

    def __init__(self, algo, percentage: float) -> None:
        super().__init__(algo)
        self.percentage = percentage

    def apply(self, alpha_model: str, price: float, direction: Direction):
        if direction == Direction.Long:
            multiplier = 1 - self.percentage
        else:
            1 + self.percentage

        sl_price = price * multiplier
        self.place_order(alpha_model, sl_price)


class FixedValueSLStrategy(AbstractSLStrategy):
    
    """
    Triggers stop loss at a fixed value above/below the current price.
    """

    def __init__(self, algo, value: float) -> None:
        super().__init__(algo)
        self.value = value

    def apply(self, alpha_model: str, price: float, direction: Direction):
        if direction == Direction.Long:
            sl_price = price - self.value
        else:
            price + self.value

        self.place_order(alpha_model, sl_price)


class TrailingStopSLStrategy(AbstractSLStrategy):
    
    """
    Adjusts the stop loss dynamically based on a 
    trailing percentage from the highest/lowest price achieved.
    """

    def __init__(self, algo, percentage: float) -> None:
        super().__init__(algo)
        self.percentage = percentage
        self.highest_price = {}
        self.lowest_price = {}

    def apply(self, alpha_model: str, price: float, direction: Direction):
        if direction == Direction.Long:
            
            self.highest_price[alpha_model] = max(
                self.highest_price.get(alpha_model, price), price
                )
            
            sl_price = self.highest_price[alpha_model] * (1 - self.percentage)
        else:

            self.lowest_price[alpha_model] = min(
                self.lowest_price.get(alpha_model, price), price
                )
            
            sl_price = self.lowest_price[alpha_model] * (1 + self.percentage)
        self.place_order(alpha_model, sl_price)