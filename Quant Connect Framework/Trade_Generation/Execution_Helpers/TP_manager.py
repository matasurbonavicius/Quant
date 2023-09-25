from AlgorithmImports import *

class TP_Manager():
    
    def __init__(self, algorithm):

        """
        Args:
            algorithm: The main algorithm instance
        """

        self.algo = algorithm

    def TP_fixed_percentage(self, 
                            percentage: float, 
                            base: float, 
                            alpha_model: float
                            ) -> float:

        """
        Here the algorithm is supposed to calculate the TP metrics

        And then sent the TP order to market

        How does the interaction with risk model work?

        Maybe this can only be called after the risk model is called /
        OR maybe the risk model calls the SL/TP itself?? But no, since we would be making
            an order before the real order

        So best is Signal -> Execution Model -> Risk Model -> Trade -> TP Manager ->
        -> Risk Model (Check if trade successful ??) -> Place TP

        High posibility of avoiding using a risk model here as it would have to be very different
        structure than it is now

        """

        pass
