import matplotlib.pyplot as plt
from typing import List, Any
import pandas as pd
import numpy as np
import warnings
import logging
import pickle
import os

from Helpers.triple_barrier_labeling import Triple_Barrier_Labeling
from Models.XGB_trainer import XGB_Trainer
from Helpers.plotting_module import PlottingModule
from Helpers.backtesting import Backtest

logger = logging.getLogger(__name__)

# Manually show python where to find my graphviz file
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'
warnings.simplefilter(action='ignore', category=FutureWarning)


class XGB_Training_Dash():
    
    """ 
    A class used to represent the XGB Training Dashboard.

    Attributes
    df_ : pd.DataFrame
        the raw dataset
    df : pd.DataFrame
        labeled dataset using Triple_Barrier_Labeling method

    Methods
    predict(data: pd.DataFrame) -> np.array:
        Makes prediction using the trained XGB model.
    """

    def __init__(self, plot: bool = True) -> None:
        
        """
        Initializes the XGB_Training_Dash class.

        Arguments:
        - Plot: if True, then plots trained model data.
        """

        self.plot = plot

        self.load_data()
        self.prepare_data()

    def load_data(self) -> None:

        """ Loads the dataset. """

        logger.info("Loading data from Data/HMM_data.csv")
        self.df_ = pd.read_csv("Data/HMM_data.csv")
        self.df = Triple_Barrier_Labeling(self.df_, 'Combined Prices').label_data()

    def save_model(
        self,
        model: Any, 
        path: str = "Models/trained_XGB_model.pkl"
        ) -> None:
        
        """
        Save the trained model to a file.

        Parameters:
        - model_trained: The trained model to be saved.
        - path: Path to save the model. Default: "models/Trained_XGB_model.pkl".
        """

        with open(path, "wb") as file:
            pickle.dump(model, file)

    def prepare_data(self) -> None:
        
        """ Prepares the training and predicting datasets. """

        logger.info("Preparing the datasets for training and predicting.")

        # Column names for training vs prediction datasets
        self.features_columns_training   = ['Dates'                    , 'Data_training_VIX'     ,   
                                            'Data_training_MOVE'       , 'Data_training_EURUSD'  , 
                                            'Data_training_HY'         , 'Data_training_BBB'     , 
                                            'Data_training_AA'         , 'label'                 ,
                                            'Data_training_Liquidity'  , 'Data_training_Spread'  , 
                                            'Data_training_FEDFUNDS'   , 'Prices_training'       ]
        
        self.features_columns_predicting = ['Dates'                    , 'Data_prediction_VIX'   , 
                                            'Data_prediction_MOVE'     , 'Data_prediction_EURUSD', 
                                            'Data_prediction_HY'       , 'Data_prediction_BBB'   , 
                                            'Data_prediction_AA'       , 'label'                 ,
                                            'Data_prediction_Liquidity', 'Data_prediction_Spread', 
                                            'Data_prediction_FEDFUNDS' , 'Prices_prediction'     ] 
        
        self.model_columns                = ['VIX'      , 'MOVE'  , 'EURUSD', 
                                             'HY'       , 'BBB'   , 'AA'    , 
                                             'Liquidity', 'Spread', 'FEDFUNDS']

        # Prepare training dataset
        self.features_train           = self.df[self.features_columns_training].dropna()
        self.dates_train              = self.features_train['Dates']
        self.target_train             = self.features_train['label']
        self.prices_train             = self.features_train['Prices_training']
        self.features_train           = self.features_train.drop(columns=[
                                                                    'Dates', 
                                                                    'label', 
                                                                    'Prices_training'
                                                                    ], axis=1)
        self.features_train.columns   = self.model_columns

        # Prepare predicting dataset
        self.features_predict         = self.df[self.features_columns_predicting].dropna()
        self.dates_predict            = self.features_predict['Dates']
        self.target_predict           = self.features_predict['label']
        self.prices_predict           = self.features_predict['Prices_prediction']
        self.features_predict         = self.features_predict.drop(columns=[
                                                                    'Dates', 
                                                                    'label', 
                                                                    'Prices_prediction'
                                                                    ], axis=1)
        self.features_predict.columns = self.model_columns
    
    def save_data(
        self,
        dates_train: pd.Series,
        features_train: pd.DataFrame,
        target_train: pd.Series,
        prices_train: pd.Series,
        dates_predict: pd.Series,
        features_predict: pd.DataFrame,
        target_predict: pd.Series,
        prices_predict: pd.Series
        ) -> None:

        """ 
        Saves the data of the trained model so 
        that it can be accessed in other modules
        
        """
        
        train_df = pd.DataFrame({
            'dates': dates_train,
            'target': target_train,
            'prices': prices_train
        })

        predict_df = pd.DataFrame({
            'dates': dates_predict,
            'target': target_predict,
            'prices': prices_predict
        })

        for x in features_train.columns:
            train_df[f'Train_{x}'] = features_train[x]

        for x in features_predict.columns:
            predict_df[f'Predict_{x}'] = features_predict[x]

        train_df.set_index('dates', inplace=True)
        predict_df.set_index('dates', inplace=True)
        final_df = pd.concat([train_df, predict_df])

        final_df.to_csv('Data/XGB_data.csv', index=False)

    def train_model(self) -> None:
        
        """ Trains the XGB model. """

        logger.info("Training the XGB model.")
        self.xgb_model = XGB_Trainer(self.features_train, self.target_train)
        self.model     = self.xgb_model.train_model()
        #self.xgb_model.plot_trained_model()

        if self.plot:
            self.xgb_model.plot_trained_model()
        
        self.save_data(self.dates_train   , self.features_train  ,
                       self.target_train  , self.prices_train    , 
                       self.dates_predict , self.features_predict, 
                       self.target_predict, self.prices_predict)
                    
        return self.model            , self.dates_train   , self.features_train, \
               self.target_train     , self.prices_train  , self.dates_predict,  \
               self.features_predict , self.target_predict, self.prices_predict 

    def predict(self, model: Any, data: pd.DataFrame) -> np.array:
        
        """ 
        Makes prediction using the trained XGB model.
        
        Parameters:
        data (pd.DataFrame): Data for which the prediction is to be made.

        Returns:
        np.array: Predicted values.
        """

        logger.info("Making predictions using the trained model.")
        return self.xgb_model.predict(model, data)

 
xgb_trainer = XGB_Training_Dash(plot = False)

model           , dates_train   , features_train , \
target_train    , prices_train  , dates_predict  , \
features_predict, target_predict, prices_predict = xgb_trainer.train_model()

xgb_trainer.save_model(model)
predicted_trained = xgb_trainer.predict(model, features_train)

minv = min(predicted_trained)
maxv = max(predicted_trained)

normalized_prediction = pd.DataFrame([(x - minv) / (maxv - minv) for x in predicted_trained])
normalized_prediction.columns = ['Normalized Values']

for index, x in normalized_prediction.iterrows():
    if x['Normalized Values'] > 0.3:
        normalized_prediction.at[index, 'Signals'] = 'green'
    elif x['Normalized Values'] < -0.3:
        normalized_prediction.at[index, 'Signals'] = 'red'
    else:
        normalized_prediction.at[index, 'Signals'] = 'yellow'

plotter = PlottingModule(dates         = dates_train, 
                         values_list   = [prices_train, normalized_prediction['Signals']], 
                         labels_list   = ['SPY', 'Signals'])
plotter.plot()

bt = Backtest(dates_train, 
              np.array(prices_train.values), 
              normalized_prediction['Signals'], 
              normalized_prediction['Signals']
              )

bt.run_backtest()
bt.plot_balance()


