from AlgorithmImports import *


class RiskModel_():

    def __init__(self, algorithm) -> None:

        """
        Args:
            algorithm: The main algorithm instance

        """

        self.algo = algorithm

    def single_position_per_model(self, alpha_model: str, trade_quantity: int, direction: InsightDirection) -> int:

        for model in self.algo.alpha_models:
            if alpha_model in model.model_name:
                current_model = model
                break
        
        if trade_quantity is None or trade_quantity == 0:
            self.algo.Debug("Trade Quantity in risk model is invalid")
            return 0

        if direction == InsightDirection.Up:
            if current_model.quantity <= 0:
                return trade_quantity
            else:
                return 0
        elif direction == InsightDirection.Down:
            if current_model.quantity >= 0:
                return trade_quantity 
            else:
                return 0

    def exit_positions_for_model(self, alpha_model: str) -> None:
        
        for model in self.algo.alpha_models:
            if alpha_model in model.model_name:
                current_model = model
                break
        
        if current_model.quantity == 0:
            return 0

        return -current_model.quantity