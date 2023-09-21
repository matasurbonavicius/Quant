from AlgorithmImports import *
from risk_model import *
from Trade_Generation.Execution_Helpers.position_sizing import *
import datetime
import pytz

class ExecutionModel_():
    
    def __init__(self, algorithm):

        """
        Args:
            algorithm: The main algorithm instance
        """

        self.algo = algorithm

    def on_insights_generated(self, algorithm: IAlgorithm, insights_collection: GeneratedInsightsCollection) -> None:
        insights = insights_collection.Insights 
        if insights:
            for insight in insights:

                if insight.Direction == InsightDirection.Up:
                    
                    risk = RiskModel_(self.algo).single_position_per_model(
                        insight.SourceModel, 
                        PositionSizing(self.algo).specified_contract_amount(10), 
                        InsightDirection.Up
                        )

                    self.algo.Log(f"{self.algo.Time} - insight up, risk: {risk}")
                    
                    if risk > 0:

                        self.algo.MarketOrder(
                            symbol = self.algo.current_symbol, 
                            quantity = risk, 
                            tag = insight.SourceModel
                            )


                if insight.Direction == InsightDirection.Down:
                    
                    risk = RiskModel_(self.algo).single_position_per_model(
                        insight.SourceModel, 
                        PositionSizing(self.algo).specified_contract_amount(-10), 
                        InsightDirection.Down
                        )

                    self.algo.Log(f"{self.algo.Time} - insight down, risk: {risk}")
                
                    if risk < 0:

                        self.algo.MarketOrder(
                            symbol = self.algo.current_symbol, 
                            quantity = risk, 
                            tag = insight.SourceModel
                            )

                if insight.Direction == InsightDirection.Flat:
                    
                    #quantity =  `   PositionSizing(self.algo).exit_positions_for_model(insight.SourceModel), 

                    self.algo.Log(f"{self.algo.Time} - insight down, risk: {risk}")
                
                    if risk < 0:

                        self.algo.MarketOrder(
                            symbol = self.algo.current_symbol, 
                            quantity = risk, 
                            tag = insight.SourceModel
                            )
        
    # --- Helper Functions ---
    
    def round_to_minimum_tick_size(self, price: float) -> float:
        properties = self.algo.Securities[self.algo.current_symbol].SymbolProperties
        tick_size = properties.MinimumPriceVariation
        return tick_size * round(price / tick_size)

    def get_LTU_timezone(self) -> str:
        algorithm_datetime = datetime.combine(self.algo.Time.date(), self.algo.Time.time())
        algorithm_timezone = pytz.timezone(str(self.algo.TimeZone))
        localized_algorithm_datetime = algorithm_timezone.localize(algorithm_datetime)
        lithuania_timezone = pytz.timezone('Europe/Lithuania')
        lithuania_datetime = localized_algorithm_datetime.astimezone(lithuania_timezone)
        return lithuania_datetime.strftime('%Y-%m-%d %H:%M')