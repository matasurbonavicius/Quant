from AlgorithmImports import *
from risk_model import *
from Trade_Generation.Execution_Helpers.position_sizing import PositionSizing
import datetime
import pytz
from Trade_Generation.Execution_Helpers.TP_manager import TakeProfitManager
from Helpers.enums import Direction

class ExecutionModel_():
    
    def __init__(self, algorithm):

        """
        Args:
            algorithm: The main algorithm instance
        """

        self.algo = algorithm

    def on_insights_generated(self, 
                              algorithm: IAlgorithm, 
                              insights_collection: GeneratedInsightsCollection
                              ) -> None:
        
        insights = insights_collection.Insights 
        current_price = self.algo.Securities[self.algo.current_symbol].Close

        if insights:
            for insight in insights:

                self.algo.Log(f"{self.algo.Time} - insight: {insight}")

                if insight.Direction == InsightDirection.Up:
                    risk = RiskModel_(self.algo).single_position_per_model(
                            insight.SourceModel, 
                            PositionSizing(self.algo).equal_qty_per_model(Direction.Long), 
                            InsightDirection.Up
                            )
                    
                    self.algo.Log(f"{self.algo.Time} - Direction long; pos sizing: {PositionSizing(self.algo).equal_qty_per_model(Direction.Long)}")
                    
                    if risk > 0:
                        self.algo.MarketOrder(
                            symbol = self.algo.current_symbol, 
                            quantity = risk, 
                            tag = insight.SourceModel
                            )
                        
                        self.algo.tp_manager.use_tp(
                            alpha_model = insight.SourceModel, 
                            price = current_price, 
                            direction = Direction.Long)
                        
                        self.algo.sl_manager.use_sl(
                            alpha_model = insight.SourceModel, 
                            price = current_price, 
                            direction = Direction.Long
                            )

                if insight.Direction == InsightDirection.Down:
                    risk = RiskModel_(self.algo).single_position_per_model(
                            insight.SourceModel, 
                            PositionSizing(self.algo).equal_qty_per_model(Direction.Short), 
                            InsightDirection.Down
                            )

                    if risk < 0:
                        self.algo.MarketOrder(
                            symbol = self.algo.current_symbol, 
                            quantity = risk, 
                            tag = insight.SourceModel
                            )
                    
                        self.algo.tp_manager.use_tp(
                                alpha_model = insight.SourceModel, 
                                price = current_price, 
                                direction = Direction.Short)
                        
                        self.algo.sl_manager.use_sl(
                                alpha_model = insight.SourceModel, 
                                price = current_price, 
                                direction = Direction.Short
                                )

                if insight.Direction == InsightDirection.Flat:
                    risk = RiskModel_(self.algo).exit_positions_for_model(
                            insight.SourceModel
                            )
                
                    if risk != 0:
                        self.algo.MarketOrder(
                            symbol = self.algo.current_symbol, 
                            quantity = risk, 
                            tag = insight.SourceModel
                            )