from QuantConnect.Algorithm import QCAlgorithm
from Trade_Generation.Execution_Helpers.TP_manager import PercentageTPStrategy
from datetime import timedelta
from AlgorithmImports import *


class AlphaModelBase(AlphaModel):
   
    def __init__(self, algorithm: QCAlgorithm, params: dict) -> None:

        """
        Initialize the UniverseSelectionModel.

        Args:
            algo: The algorithm instance.
        """

        self.algo = algorithm
        
        """ALPHA MODEL STATUS VARIABLES"""
                
        self.params = params # Parameters passed to the alpha model       
        self.symbol = "" # Symbol currently traded
        self.quantity = 0 # Quantity currently held
        self.average_price = None # Average price of the current position

        # Unique Alpha Model name
        self.model_name = ""
        for key, value in self.params.items():
            self.model_name += f"{value} "

        self.generated_insights = []
        self.Schedulers()
        self.Indicators()

        self.level = None
        self.period_end = False

        tp_strategy = PercentageTPStrategy(self.algo, 0.05)
        self.algo.tp_manager.register_tp_strategy(self.model_name, tp_strategy)


    # -----------------------------------------------------------------------------

    def OnSecuritiesChanged(self, algorithm: QCAlgorithm, changes: dict) -> None:
        for security in changes.AddedSecurities:
            if security.Symbol.Value == self.algo.signal_instrument.Symbol:
                
                self.cons = TradeBarConsolidator(timedelta(minutes=self.params["Timeframe"]))
                self.cons.DataConsolidated += self.Consolidated_Update
                algorithm.SubscriptionManager.AddConsolidator(security.Symbol, self.cons)

    def Status(self, orderEvent: object) -> None:

        # Update model status variables
        self.quantity += orderEvent.FillQuantity
        self.symbol = orderEvent.Symbol
        #self.average_price = orderEvent.AverageFillPrice
        
        if orderEvent.Direction == OrderDirection.Sell:
            self.algo.Plot("Main Chart", 
                           "Sell", 
                           orderEvent.FillPrice)

        if orderEvent.Direction == OrderDirection.Buy:
            self.algo.Plot("Main Chart", 
                           "Buy", 
                           orderEvent.FillPrice)

    def Schedulers(self) -> None:

        symbol = self.algo.signal_instrument.Symbol

        date_rule_end = {
            "Month": self.algo.DateRules.MonthEnd(symbol),
            "Week": self.algo.DateRules.WeekEnd(symbol),
            "Day": self.algo.DateRules.EveryDay(symbol)
            }

        date_rule_start = {
            "Month": self.algo.DateRules.MonthStart(symbol),
            "Week": self.algo.DateRules.WeekStart(symbol),
            "Day": self.algo.DateRules.EveryDay(symbol)
            }
        
        self.algo.Schedule.On(
            date_rule_end[self.params["Time_rule"]], 
            self.algo.TimeRules.BeforeMarketClose(symbol, 2), 
            Action(lambda: self.end_period(self.params["Time_rule"]))
            )
    
        self.algo.Schedule.On(
            date_rule_start[self.params["Time_rule"]], 
            self.algo.TimeRules.AfterMarketOpen(symbol, 0), 
            Action(lambda: self.start_period(self.params["Time_rule"]))
            )    

    def Indicators(self) -> None:

        # Add a Moving Average
        MA_Consolidator = self.algo.ResolveConsolidator(
            self.algo.signal_instrument.Symbol, 
            timedelta(minutes=self.params["Timeframe"])
            )

        self.algo.SubscriptionManager.AddConsolidator(
            self.algo.signal_instrument.Symbol, 
            MA_Consolidator
            )

        self.MA = SimpleMovingAverage(
            self.algo.signal_instrument.Symbol, 
            self.params["MAPeriod"]
            )

        self.algo.RegisterIndicator(
            self.algo.signal_instrument.Symbol, 
            self.MA, 
            MA_Consolidator
            )

        self.ma_window = RollingWindow[float](2)

    # -----------------------------------------------------------------------------

    def Update(self, algorithm: QCAlgorithm, data: object) -> list:  
        insights = self.generated_insights
        self.generated_insights = []

        if self.period_end == True:
            self.period_end = False
            insights.append(self.insight(InsightDirection.Flat))

        return insights

    def Consolidated_Update(self, sender: Any, bar: object) -> None: 
        self.ma_window.Add(self.MA.Current.Value)
        if self.algo.warmup_finished:
            self.algo.Plot("Main Chart", "Symbol", bar.Close)
            self.algo.Plot("Main Chart", "Level", self.level)

            if bar.Close > self.level:

                self.generated_insights.append(
                    self.insight(InsightDirection.Up))
                    
    # -----------------------------------------------------------------------------
    
    def insight(self, direction: int) -> object:
        
        insight = Insight(
            self.algo.signal_instrument.Symbol, 
            timedelta(minutes=1), 
            InsightType.Price, 
            direction
            )
        
        insight.SourceModel = self.model_name
        return insight

    def end_period(self, period) -> None:
        
        """
        Scheduled Event
        """

        self.period_end = True

    def start_period(self, period) -> None: 
        
        """
        Scheduled Event
        """

        self.level = self.algo.Securities[
            self.algo.signal_instrument.Symbol
            ].Open