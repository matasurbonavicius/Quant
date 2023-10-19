from typing import Tuple, List
import matplotlib.pyplot as plt
from collections import namedtuple
import pandas as pd
import numpy as np
import pickle

from Models.DTW_classification import classify_states
from Models.BIC_calculator import HMMCalculatorBIC
from Models.HMM_trainer import HMMTrainer

class MSCM():

    """
    #------------------------------------------------------------------------------
    Market State Classification Module

    This module provides a systematic approach to classifying market states 
    using a Hidden Markov Model (HMM). The process is divided into several steps:

    1. Initialization: Sets up logging configurations for tracking of operations.
    2. Data Acquisition: Fetches financial and market data from FRED and yFinance.
    3. Data Processing: Transforms the raw data by computing Z-scores, handling 
    missing values, and scaling. The data is preprocessed to be used 
    effectively by the HMM.
    4. Model Training and Prediction: Trains an HMM on the processed data, 
    saves the trained model, and predicts market states for a given dataset.
    5. Visualization: Plots the predicted market states and other relevant 
    financial indicators.
    6. Backtesting: Uses the predicted market states to simulate a trading 
    strategy and visualizes the resulting balance over time.

    The objective is to classify different market states (bull, bear, neutral) 
    based on historical financial indicators and use these predictions to make 
    informed trading decisions.

    Data that is needed for the market stress indicator:

    market_stress_indicator = {
        "US Equity": "^VIX",            # Volatility  -- yFinance
        "US Bonds": "^MOVE",            # Volatility  -- yFinance
        "Global FX": "EURUSD=X ATR",    # Volatility  -- yFinance -- ATR on EURUSD
        "HY Spread": "BAMLH0A0HYM2",    # Spreads HY  -- Fred
        "BBB Spread": "BAMLC0A4CBBB",   # Spreads BBB -- Fred
        "AA Spread": "BAMLC0A2CAA",     # Spreads AA  -- Fred
        "SPX Liquidity": "Liquidity",   # Liquidity   -- yFinance -- "SPY"
        "Yield Curve": "T10Y2Y",        # Liquidity   -- Fred
        "FED FUNDS Rate": "FEDFUNDS"    # Liquidity   -- Fred     -- "FEDFUNDS"
    }
    weights = [1,1,1,1,1,1,1,1,1]

    Parameters:
    - tr_data: start date & end date of the training period
    - pr_data: start date & end date of the prediction period

    #------------------------------------------------------------------------------
    """

    def __init__(
            self, 
            tr_data: list = ['2005-01-01', '2015-01-01'], 
            pr_data: list = ['2015-01-01', '2023-01-01']
            ) -> None:

        self.ProcessedData = namedtuple('ProcessedData', [
                'training_data_Z', 
                'prediction_data_Z', 
                'prices_training', 
                'prices_prediction', 
                'training_dates', 
                'prediction_dates'
            ])
        
        self.tr_data = tr_data
        self.pr_data = pr_data

    def get_api_key(self) -> str:
        """
        Retrieve the API key for fetching data.

        Returns:
        - str: API key.
        """
        return "b48eef00ef6d0de74f782e6165ac7454" 

    def perform_BIC(
        self,
        training_data: pd.DataFrame, 
        n_train: int, 
        max_n_components: int
        ) -> None:
        
        """
        Calculate and plot the BIC for the given training data.

        Parameters:
        - training_data: The data to be used for the BIC calculation.
        - n_train: Number of trainings to be performed.
        - max_n_components: Maximum number of components for the BIC calculation.
        """

        # Perform the BIC calculation, get average BIC line
        data = HMMCalculatorBIC(n_train, max_n_components).calculate(training_data)

        # Print BIC. Currently 2 States is optimal
        plt.plot(data.reshape(-1,1)[:len(data)-1])
        plt.show()

    def save_model(
        self,
        model_trained: object, 
        path: str = "models/trained_HMM_model.pkl"
        ) -> None:
        
        """
        Save the trained model to a file.

        Parameters:
        - model_trained: The trained model to be saved.
        - path: Path to save the model. Default: "models/trained_model.pkl".
        """

        with open(path, "wb") as file:
            pickle.dump(model_trained, file)

    def process_data(
        self,
        data: pd.DataFrame, 
        data_module: object
        ) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
            pd.DatetimeIndex, pd.DatetimeIndex
            ]:
        
        """
        Process raw data to prepare it for training and prediction.

        Parameters:
        - data: Raw data as pandas DataFrame.
        - data_module: Data module to assist in processing.

        Returns:
        - Tuple: Processed data arrays and date indices.
        """

        feature_columns = [
        "Close_^VIX", "Close_^MOVE", "ATR_EURUSD=X",             # Volatility
        "BAMLH0A0HYM2", "BAMLC0A4CBBB", "BAMLC0A2CAA",           # Spreads
        "Liquidity_Indicator_Log_EMA_SPY", "T10Y2Y", "FEDFUNDS"  # Liquidity
        ]

        training_data    = data[self.tr_data[0]:self.tr_data[1]][feature_columns]
        prediction_data  = data[self.pr_data[0]:self.pr_data[1]][feature_columns]

        window = 30

        # Compute rolling mean and standard deviation
        roll_means_training = training_data.rolling(window=window).mean()
        roll_means_prediction = prediction_data.rolling(window=window).mean()

        roll_std_devs_training = training_data.rolling(window=window).std()
        roll_std_devs_prediction = prediction_data.rolling(window=window).std()

        # Compute Z-scores
        training_data_Z = (
            (training_data - roll_means_training) / roll_std_devs_training
            )
        prediction_data_Z = (
            (prediction_data - roll_means_prediction) / roll_std_devs_prediction
            )

        training_data_Z = training_data_Z.ffill()
        prediction_data_Z = prediction_data_Z.ffill()

        # Before scaling align prices with training_data and prediction_data
        training_dates = training_data_Z.dropna().index
        prediction_dates = prediction_data_Z.dropna().index

        # Filter SPY_close_prices using these indices
        prices_training = data.loc[training_dates, "Close_SPY"].values
        prices_prediction = data.loc[prediction_dates, "Close_SPY"].values

        # Scale data
        training_data_Z, _ = data_module.scale_data(training_data_Z)
        prediction_data_Z, _ = data_module.scale_data(prediction_data_Z)

        # Scale the aligned prices arrays
        prices_training, _ = data_module.scale_data(prices_training, True)
        prices_prediction, _ = data_module.scale_data(prices_prediction, True)

        return self.ProcessedData(
            training_data_Z, 
            prediction_data_Z, 
            prices_training, 
            prices_prediction, 
            training_dates, 
            prediction_dates
        )

    def train_model(
        self,
        training_data_Z: np.ndarray, 
        prediction_data_Z: np.ndarray, 
        prices_training: np.ndarray
        ) -> Tuple[List[str], np.ndarray, dict]:
        
        """
        Train the HMM model n_train number of times, choose the 
        best model and using it and classify market states.

        Parameters:
        - training_data_Z: Training data.
        - prediction_data_Z: Prediction data.
        - prices_training: SPY close prices for training.

        Returns:
        - Tuple: Colors for plotting, predicted values, and state classifications.
        """

        HMM               = HMMTrainer(training_data_Z)
        model_trained     = HMM.train_n_models(n_train=10, n_components=6)
        self.save_model(model_trained)

        predicted_values  = HMM.predict(model_trained, prediction_data_Z)
        trained_values    = HMM.predict(model_trained, training_data_Z)

        states_classified = classify_states(prices   = prices_training.flatten(), 
                                            states   = trained_values, 
                                            n_states = 6)
        colors            = [states_classified[state] for state in predicted_values]

        return colors, predicted_values, trained_values, states_classified

    def save_data(
        self,
        prediction_data_Z: np.ndarray, 
        prices_prediction: np.ndarray, 
        prediction_dates: np.ndarray,
        training_data_Z: np.ndarray, 
        prices_training: np.ndarray, 
        training_dates: np.ndarray,
        labels_list: list,
        predicted_values: list,
        trained_values: list
        ) -> None:
        
        """
        Saves data that is necessary for prediction and training,
        so that other models can use the classified data. Data is saved as
        a CSV.

        Parameters:
        - prediction_data_Z: Data used for prediction.
        - prices_prediction: Prices corresponding to the prediction data.
        - prediction_dates: Dates corresponding to the prediction data.
        - training_data_Z: Data used for training.
        - prices_training: SPY close prices for training.
        - training_dates: Dates corresponding to the training data.

        Returns:
        - None. File is saved in the specified directory.
        """

        df_training = pd.DataFrame({
            'Dates': training_dates,
            'Prices_training': prices_training.flatten(),
            'trained_values': trained_values
        })

        df_prediction = pd.DataFrame({
            'Dates': prediction_dates,
            'Prices_prediction': prices_prediction.flatten(),
            'Predicted_values': predicted_values
        })

        for i, label in enumerate(labels_list):
            df_training[f'Data_training_{label}'] = training_data_Z[:, i]

        for i, label in enumerate(labels_list):
            df_prediction[f'Data_prediction_{label}'] = prediction_data_Z[:, i]

        df = pd.merge(
            df_training, 
            df_prediction, 
            on='Dates', 
            how='outer'
            ).sort_values(by='Dates')

        pt, pp = 'Prices_training', 'Prices_prediction'
        df['Combined Prices'] = df[pt].combine_first(df[pp])
        # df = df.drop([pt, pp], axis=1)

        df.to_csv('Data/training_prediction_data.csv', index=False)




