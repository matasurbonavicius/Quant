from AlgorithmImports import *
from Universe.securities_manager import *
from Trade_Generation.execution_model import *
from Universe.universe_model import *
from Status_Control.status_manager import *
from Helpers.trades_export import *
from Trade_Generation.alpha_model import *
from Helpers.charting import *


class MyAlgorithm(QCAlgorithm):
    
    def Initialize(self) -> None:
        
        # -----------------------------------------------------------------------------
        # PARAMETERS
        
        """ SELECT DATES AND INITIAL CASH """
        self.SetTimeZone("America/New_York")
        self.SetStartDate(2023, 5, 1)
        self.SetEndDate(2023, 9, 5) 
        self.SetCash(10000000)
       
        """ SET ALGORITHM WARM UP PERIOD """
        self.SetWarmup(timedelta(days=60))

        """ SELECT INSTRUMENTS """
        UniverseModel(self).one_instrument_model(
            instrument = self.AddEquity(
            ticker = "SPY", 
            resolution = Resolution.Minute
            )
            )
        
        """ 
        CREATE A LIST OF ALPHA MODELS
            - Give alpha models parameters
            - AddAlpha function adds model to the algorithm
        """
        self.alpha_models = []
        for timeframe in [30]:
                parameters = {
                    "Timeframe": timeframe,
                    "MAPeriod": 20,
                    "Time_rule": "Day"
                    }
                alpha_model = AlphaModelBase(self, parameters)
                self.alpha_models.append(alpha_model)
                self.AddAlpha(alpha_model)
   
        """ SET BROKERAGE MODEL """
        self.SetBrokerageModel(BrokerageName.QuantConnectBrokerage, AccountType.Cash)

        """ INITIALIZE CHARTS """
        Charting(self, "Main Chart")
        Charting(self, "Secondary Chart")
        
        # ---
        self.InsightsGenerated += self.OnInsightsGenerated
        self.init_helpers()
        self.execution_model = ExecutionModel_(self)

        # -----------------------------------------------------------------------------
    
    def OnSecuritiesChanged(self, changes: object) -> None:

        # Keeps securities list updated if we are subscribed to any futures
        new = [x for x in changes.AddedSecurities if x.Symbol.SecurityType == SecurityType.Future]
        old = [x for x in changes.RemovedSecurities if x.Symbol.SecurityType == SecurityType.Future]
        if new:
            self.universe_symbols.append(sorted(new, key=lambda x: x.Expiry, reverse=True)[0])
        if old:
            self.universe_symbols.remove(old)

    # Called every data point
    def OnData(self, data) -> None:

        # Call securities manager on first data point and then every week
        if self.on_data_first_call or self.week_start:
            self.current_symbol = SecuritiesManager(self).check_universe(data)
            self.on_data_first_call, self.week_start = False, False
    
    def OnInsightsGenerated(self, 
                            algorithm: IAlgorithm, 
                            insights_collection: GeneratedInsightsCollection
                            ) -> None:

        self.execution_model.on_insights_generated(algorithm, insights_collection)
    
    def OnOrderEvent(self, orderEvent: object) -> None:
        StatusManager(self, orderEvent)
                    
    def OnWarmupFinished(self) -> None:
        self.warmup_finished = True
    
    def OnEndOfAlgorithm(self) -> dict:
        TradesExport(self).print_all_trades()
    
    # --- Helper functions ---

    def init_helpers(self) -> None:
        
        # Scheduler to call Securities Manager every week
        self.Schedule.On(
            self.DateRules.WeekStart(self.signal_instrument.Symbol), 
            self.TimeRules.AfterMarketOpen(self.signal_instrument.Symbol, 0), 
            self.WeekStart
            )

        self.on_data_first_call = True
        self.warmup_finished = False
        self.week_start = False

    def WeekStart(self) -> bool:
        self.week_start = True
        return self.week_start
    
