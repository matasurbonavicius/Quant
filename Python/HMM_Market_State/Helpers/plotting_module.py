import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from typing import List, Optional, Union


class PlottingModule:
    
    """
    Advanced Plotter for visualizing data.

    Allows for plotting multiple value lists against a common date axis.

    Attributes:
        dates: A sequence of dates.
        values_list: A list of value lists.
        labels_list: A list of labels.
        color_mapping: Optional color mapping for scatter plots.
    """

    def __init__(self, 
                 dates:         Union[np.ndarray, pd.DataFrame, pd.Series], 
                 values_list:   List[Union[np.ndarray, pd.DataFrame, pd.Series]], 
                 labels_list:   List[str], 
                 color_mapping: Optional[str] = None
                 ) -> None:
        
        self.dates = self._to_numpy_array(dates)
        self.values_list = [self._to_numpy_array(values) for values in values_list]
        self.labels_list = labels_list
        self.color_mapping = color_mapping
        
        # Validate data
        if len(self.values_list) != len(self.labels_list):
            logging.error("Mismatch between number of value lists and labels.")
            raise ValueError("values_list and labels_list must be of the same length.")

    @staticmethod
    def _to_numpy_array(data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """ Convert data to numpy array if it isn't already. """

        if isinstance(data, (pd.DataFrame, pd.Series, pd.DatetimeIndex)):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            logging.error(
                "Invalid data type provided. Expected pd.DataFrame, pd.Series, or np.ndarray."
                )
            raise TypeError("Invalid data type provided.")

    def plot(self) -> None:
        """ Plot the provided data. """
        
        fig = plt.figure(figsize=(14, 10))
        
        # Define the relative heights
        heights = [4] + [1] * (len(self.values_list) - 1)
        gs = gridspec.GridSpec(len(self.values_list), 1, height_ratios=heights)
        
        for i, (values, label) in enumerate(zip(self.values_list, self.labels_list)):
            ax = plt.subplot(gs[i])
            
            if self.color_mapping:
                ax.scatter(self.dates, values, color=self.color_mapping, s=10, label=label)
            else:
                ax.plot(self.dates, values, label=label)
            
            ax.set_ylabel(label, color='black')
            ax.tick_params(axis='y', labelcolor='black')
            ax.legend(loc="upper left")
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()

