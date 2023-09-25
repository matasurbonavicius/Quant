from AlgorithmImports import *
from Helpers.enums import Direction

class AbstractTPStrategy:

    """
    Base class for take profit strategies.
    """

    def __init__(self, algo) -> None:
        self.algo = algo

    def apply(self, alpha_model: str, price: float, direction: Direction):

        """
        Apply the take profit strategy.
        """

        raise NotImplementedError

    def place_order(self, alpha_model: str, tp_price: float):

        """
        Places a limit order for the take profit strategy.
        """

        for model in self.algo.alpha_models:
            if alpha_model in model.model_name:
                current_model = model
                break

        self.algo.LimitOrder(
            symbol=self.algo.signal_instrument.Symbol,
            quantity=-current_model.quantity,
            limitPrice=tp_price,
            tag=f"{current_model} - Take Profit"
        )


class TakeProfitManager:

    """
    Manages take profit strategies for alpha models.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(TakeProfitManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, algorithm) -> None:
        self.algo = algorithm
        self.alpha_models_tp = {}

    def register_tp_strategy(self, alpha_model: str, strategy: AbstractTPStrategy):

        """
        Register a take profit strategy for an alpha model.
        """

        self.alpha_models_tp[alpha_model] = strategy
        self.algo.Debug(f"Registered TP Strategy for {alpha_model}")

    def use_tp(self, alpha_model, price, direction: Direction) -> None:

        """
        Use the registered take profit strategy.
        """

        strategy = self.alpha_models_tp.get(alpha_model)
        if strategy:
            strategy.apply(alpha_model, price, direction)


class PercentageTPStrategy(AbstractTPStrategy):

    """
    Takes profit based on a percentage of the current price.
    """

    def __init__(self, algo, percentage: float) -> None:
        super().__init__(algo)
        self.percentage = percentage

    def apply(self, alpha_model: str, price: float, direction: Direction):
        if direction == Direction.Long:
            multiplier = 1 + self.percentage
        else:
            1 - self.percentage

        tp_price = self.round_to_minimum_tick_size(price * multiplier)
        self.place_order(alpha_model, tp_price)

    def round_to_minimum_tick_size(self, price: float) -> float:
        properties = self.algo.Securities[
            self.algo.signal_instrument.Symbol
            ].SymbolProperties
        tick_size = properties.MinimumPriceVariation
        return tick_size * round(price / tick_size)


class FixedValueTPStrategy(AbstractTPStrategy):

    """
    Takes profit at a fixed value above/below the current price.
    """

    def __init__(self, algo, value: float) -> None:
        super().__init__(algo)
        self.value = value

    def apply(self, alpha_model: str, price: float, direction: Direction):
        if direction == Direction.Long:
            tp_price = price + self.value
        else:
            price - self.value

        self.place_order(alpha_model, tp_price)


class TrailingStopTPStrategy(AbstractTPStrategy):

    """
    Adjusts the take profit dynamically based on a 
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
            
            tp_price = self.highest_price[alpha_model] * (1 - self.percentage)
        else:
            
            self.lowest_price[alpha_model] = min(
                self.lowest_price.get(alpha_model, price), price
                )
            
            tp_price = self.lowest_price[alpha_model] * (1 + self.percentage)
        self.place_order(alpha_model, tp_price)