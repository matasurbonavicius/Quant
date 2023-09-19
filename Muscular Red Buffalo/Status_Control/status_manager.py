from AlgorithmImports import *
from Trade_Generation.alpha_model import *


class StatusManager():
        
    def __init__(self, algorithm, orderEvent):

        """
        Args:
            algo: The algorithm instance.
            orderEvent: any order event that 
        """

        self.algo = algorithm
        self.order = self.algo.Transactions.GetOrderById(orderEvent.OrderId)
        
        self.route_status(orderEvent)

    def route_status(self, orderEvent: OrderEvent) -> None:
        for model in self.alpha_models:
            if model.model_name in self.order.Tag:
                if self.order.Status == OrderStatus.Filled:
                    model.Status(orderEvent)
