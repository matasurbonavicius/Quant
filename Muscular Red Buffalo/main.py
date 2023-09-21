from Trade_Generation.execution_model import *
from Status_Control.status_manager import *
from Trade_Generation.alpha_model import *
from QuantConnect.Orders import Direction
from Universe.securities_manager import *
from Universe.universe_model import *
from Helpers.charting import *
from AlgorithmImports import *


class MyAlgorithm(QCAlgorithm):
    
    def Initialize(self) -> None:
        
        """
        Initialize function is the first function to be called, thus is the place
            for all the settings and parameters

            This function oversees the creation of Universe Model and Alpha Models

            Also, this function sets the main parameters to the Risk Model, which 
                set the behavioral of it
        """
        
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
                for direction in [Direction.Up, Direction.Down]:
                    parameters = {
                        "Timeframe": timeframe,
                        "MAPeriod": 20,
                        "Time_rule": "Day",
                        "Direction": direction
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

    
    def OnSecuritiesChanged(self, changes: object) -> None:

        """
        Function is called every time a new security is added/deleted/expired/etc. 
        This is the place to keep the futures management which keeps our list of 
        currently traded futures. 

        We then call Securities Manager and use this list to choose the most 
        important future contract
        
        """

        new = []
        for x in changes.AddedSecurities:
            if x.Symbol.SecurityType == SecurityType.Future:
                new.append(x)

        old = []
        for x in changes.RemovedSecurities:
            if x.Symbol.SecurityType == SecurityType.Future:
                old.append(x)
        
        if new:
            self.universe_symbols.append(sorted(new, key=lambda x: x.Expiry, reverse=True)[0])
        if old:
            self.universe_symbols.remove(old)

    def OnData(self, data) -> None:

        """
        Function that is called every smallest data point

        Main function that is placed here is a check of securities every
        week. 
        - This is needed in case we are trading futures. We want to 
            rotate the futures, keep them the latest.
        
        """

        # Call securities manager on first data point and then every week
        if self.on_data_first_call or self.week_start:
            self.current_symbol = SecuritiesManager(self).check_universe(data)
            self.on_data_first_call, self.week_start = False, False
    
    def OnInsightsGenerated(self, 
                            algorithm: IAlgorithm, 
                            insights_collection: GeneratedInsightsCollection
                            ) -> None:
        
        """
        Insights = Signals. This is internal Quant Connect language

        Insights are received in OnInsightsGenerated function which I
            then pass on to the Execution Model

            Yes, OnInsightsGenerated is essentially the Execution Model
        """

        self.execution_model.on_insights_generated(algorithm, insights_collection)
    
    def OnOrderEvent(self, orderEvent: object) -> None:

        """
        Status Manager is called which will then redirect orderEvent to
            the corresponding alpha model
        """

        StatusManager(self, orderEvent)
                    
    def OnWarmupFinished(self) -> None:

        """
        Function that fires off once per algorithm indicating
            that warm up has finished and the algorithm can start trading
        """

        self.warmup_finished = True
    
    # --- Helper functions ---

    def init_helpers(self) -> None:
        
        """
        Every function that is a helper and does not belong on
            a fancy palace like the Initialize function of MyAlgorithm
            should end up here. Also:

            - Schedulers
            - Variables
        """

        self.Schedule.On(
            self.DateRules.WeekStart(self.signal_instrument.Symbol), 
            self.TimeRules.AfterMarketOpen(self.signal_instrument.Symbol, 0), 
            self.WeekStart
            )

        self.on_data_first_call = True
        self.warmup_finished = False
        self.week_start = False

    def WeekStart(self) -> bool:

        """
        A helper function created in def init_helpers
            its use: to set a variable self.week_start
            to be equal to True at the start of the week
            so we can call the Securities Manager
        """

        self.week_start = True
        return self.week_start
    
