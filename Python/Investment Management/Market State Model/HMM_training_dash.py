import pandas as pd
import numpy as np
import logging
import warnings

from Data.market_data_module import MarketDataModule
from Helpers.plotting_module import PlottingModule
from Helpers.backtesting import Backtest
from Models.MSCM import MSCM
from Models.DTW_classification import classify_states


class HMM_Training_Dash:

    """
    This class serves as a dashboard for market data training and analysis.
    
    Arguments & main attributes:
        - tr_data (list): Date range for training data.
        - pr_data (list): Date range for prediction data.
        - symbols_fred (list): List of symbols to fetch from FRED.
        - symbols_yfinance (list): List of symbols to fetch from Yahoo Finance.
        - perform_bic (bool): Flag to determine whether BIC analysis should be performed.
        - run_backtest (bool): Flag to determine whether backtesting should be performed.
        - plotting (bool): Flag to determine whether plotting should be done.
        
    """

    def __init__(
            self, 
            tr_data: list = ["2005-01-01", "2015-01-01"], 
            pr_data: list = ["2015-01-01", "2023-01-01"],
            perform_bic: bool = True,
            run_backtest: bool = True,
            plotting: bool = True
            ) -> None:
        
        self.symbols_fred     = ["BAMLH0A0HYM2", "BAMLC0A4CBBB", "BAMLC0A2CAA", 
                                 "T10Y2Y", "FEDFUNDS", "DTB3"]
        self.symbols_yfinance = ["^VIX", "^MOVE", "EURUSD=X", "SPY"]
        
        self.data_module = MarketDataModule(symbols_fred=self.symbols_fred, 
                                            symbols_yfinance=self.symbols_yfinance)
        self.MSCM = MSCM(tr_data, pr_data)
        
        self.data = None
        self.processed_data = None
        self.tr_data = tr_data
        self.pr_data = pr_data
        self.perform_bic = perform_bic
        self.run_backtest = run_backtest
        self.plotting = plotting

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

    def perform_analysis(self) -> None:
        training_data_Z   = self.processed_data.training_data_Z
        prediction_data_Z = self.processed_data.prediction_data_Z
        prices_training   = self.processed_data.prices_training
        prices_prediction = self.processed_data.prices_prediction
        training_dates    = self.processed_data.training_dates
        prediction_dates  = self.processed_data.prediction_dates

        labels_list=["VIX", "MOVE", "EURUSD", "HY", 
                     'BBB', 'AA', 'Liquidity', 'Spread', 'FEDFUNDS']

        if self.perform_bic:
            self.MSCM.perform_BIC(training_data_Z, 5, 10)

        if self.plotting:
            predicted_values, trained_values = self.MSCM.train_model(training_data_Z, 
                                                                    prediction_data_Z, 
                                                                    prices_training)
            
            classified_trained = classify_states(prices   = prices_training.flatten(), 
                                                 states   = trained_values, 
                                                 n_states = 6)
            colors_predict     = [classified_trained[state] for state in predicted_values]
            
            self.MSCM.save_data(
                prediction_data_Z, 
                prices_prediction, 
                prediction_dates,
                training_data_Z, 
                prices_training, 
                training_dates,
                labels_list,
                predicted_values,
                trained_values
                )

            values_list_efficient = [prices_prediction] + [prediction_data_Z[:, i] for i in range(9)]
            plotter = PlottingModule(dates         = prediction_dates, 
                                    values_list   = values_list_efficient, 
                                    labels_list   = ['SPY', 'VIX', 'MOVE', 'EURUSD', 'HY', 
                                                     'BBB', 'AA', 'Liquidity', 'Spread', 'FEDFUNDS'], 
                                    color_mapping = colors_predict)
            plotter.plot()

        if self.run_backtest:
            bt = Backtest(prediction_dates, 
                        np.array(self.data.loc[prediction_dates, "Close_SPY"].values), 
                        predicted_values, 
                        classified_trained
                        )
            bt.run_backtest()
            bt.plot_balance()

    def run(self):
        self.initialize_logging()
        self.fetch_data()
        self.calculate_indicators()
        self.preprocess_data()
        self.perform_analysis()

dash = HMM_Training_Dash(perform_bic = False)
dash.run_backtest = True
dash.run()