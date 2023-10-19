import pandas as pd
import numpy as np
import logging
import warnings
import pickle

from Data.market_data_module import MarketDataModule
from Helpers.plotting_module import PlottingModule
from Helpers.backtesting import Backtest
from Models.MSCM import MSCM
from Models.HMM_trainer import HMMTrainer


class HMM_Current_State_Dash:
    def __init__(
        self, 
        tr_data: list = ["2005-01-01", "2015-01-01"], 
        pr_data: list = ["2015-01-01", "2023-01-01"],
        ) -> None:
        
        self.symbols_fred     = ["BAMLH0A0HYM2", "BAMLC0A4CBBB", "BAMLC0A2CAA", 
                                 "T10Y2Y", "FEDFUNDS", "DTB3"]
        self.symbols_yfinance = ["^VIX", "^MOVE", "EURUSD=X", "SPY"]
        self.data_module = MarketDataModule(symbols_fred=self.symbols_fred, 
                                            symbols_yfinance=self.symbols_yfinance)
        
        self.MSCM = MSCM()
        self.MODEL = HMMTrainer(np.zeros(10))
        
        self.data = None
        self.processed_data = None
        self.tr_data = tr_data
        self.pr_data = pr_data

    def initialize_logging(self) -> None:
        
        logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        warnings.simplefilter(action='ignore', category=FutureWarning)

    def fetch_data(self) -> None:
        self.data_module.fetch_data_fred(self.MSCM.get_api_key(), self.tr_data[0])
        self.data_module.fetch_data_yfinance(self.tr_data[0])

    def calculate_indicators(self) -> None:
        self.data_module.calculate_indicators(['ATR'], "Close_EURUSD=X")
        self.data_module.calculate_indicators(['Liquidity'], "Close_SPY")
        self.data_module.calculate_indicators(['Spread'], symbol1="FEDFUNDS", symbol2="DTB3")
        self.data = self.data_module.get_data().ffill()

    def preprocess_data(self) -> None:
        self.processed_data = self.MSCM.process_data(self.data, self.data_module)
    
    def get_latest_model(self, model_path="Models/trained_HMM_model.pkl") -> None:
        with open(model_path, "rb") as file:
            self.model = pickle.load(file)
    
    def get_state_classification(self, state: int) -> str:

        df = pd.read_csv('Models/state_classification.csv')
        state = str(state[0])

        return df[state].values, state

    def predict_current_state(self) -> None:
        prediction_data_Z = self.processed_data.prediction_data_Z[-1].reshape(1, -1)
        prediction = self.model.predict(prediction_data_Z)
        state, state_no = self.get_state_classification(prediction)
        next_state, _ = self.get_state_classification([np.argmax(self.model.transmat_[int(state_no)])])
        
        # Some formatting of the terminal window
        for x in range(10):
            print("-")
        print("---------------------------------------")
        print(f"CURRENT STATE OF THE MARKET: {state}")
        print(f"NEXT MOST LIKELY STATE OF THE MARKET: {next_state}")
        print("---------------------------------------")

    def run(self):
        self.initialize_logging()
        self.fetch_data()
        self.calculate_indicators()
        self.preprocess_data()
        self.get_latest_model()
        self.predict_current_state()

dash = HMM_Current_State_Dash()
dash.run()