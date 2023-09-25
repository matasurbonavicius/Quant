from AlgorithmImports import *


class RiskModel_():

    def __init__(self, algorithm) -> None:

        """
        Args:
            algorithm: The main algorithm instance

        Note:

            input: modelis ir siulomas naujas treidas
            output: max kiekis kiek leidziama pirkti (gali buti nulis)

            jei position sizing sako perkam 5 lotus, o rizikos modelis sako
            max yra 3, tai imam 3

        """

        self.algo = algorithm

    def single_position_per_model(self, alpha_model: str, trade_quantity: int, direction: InsightDirection) -> int:

        for model in self.algo.alpha_models:
            if alpha_model in model.model_name:
                current_model = model
                break
        
        if trade_quantity is None or trade_quantity == 0:
            self.algo.Debug("Trade Quantity in risk model is invalid")

        self.algo.Log(f"{self.algo.Time} - Risk Model, model quantity: {current_model.quantity}")

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