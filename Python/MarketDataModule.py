from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Set
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MarketDataModule:
    
    """
    A class to fetch market data and calculate various indicators, manipulate data.
    """

    def __init__(self, symbols: List[str]) -> None:
        
        self.current_indicators_names: Set[str] = set()
        self.dataframes: Optional[pd.DataFrame] = None
        
        self.symbols: List[str] = symbols
        self.indicators: dict = {
            'RSI': self._compute_rsi,
            'CCI': self._compute_cci,
            'ATR': self._compute_atr,
            'MA': self._compute_ma,
            'EMA': self._compute_ema,
            'Stoch': self._compute_stochastic,
            'BB': self._compute_bollinger_bands
        }

    def fetch_data(self, start_date: str, end_date: str = None) -> None:
        
        """
        Fetch data for the given symbols.
        """

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        for idx, symbol in enumerate(self.symbols):
            data = yf.download(
                symbol, 
                start=start_date, 
                end=end_date, 
                interval='1wk', 
                progress=False
            )

            data.rename(columns=lambda x: f"{x}_{symbol}", inplace=True)

            if self.dataframes is None:
                self.dataframes = data
            else:
                self.dataframes = pd.merge(self.dataframes, 
                                           data, 
                                           left_index=True, 
                                           right_index=True, 
                                           how='inner'
                                           )

    def get_data(self) -> pd.DataFrame:
        
        """
        Return the self.dataframes dataframe.
        """

        return self.dataframes
    
    def get_indicators(self) -> List[str]:
        
        """
        Return the names of the currently computed indicators.
        """

        return list(self.current_indicators_names)
    
    def scale_data(self, dataframe_: pd.DataFrame) -> np.ndarray:
        
        """
        Drops na's & Scales data
        Returns np array with mean of 0 & st. Dev 1
        """

        dataframe = dataframe_.dropna(inplace=True)
        dataframe = StandardScaler().fit_transform(dataframe_)
        return dataframe

    def calculate_indicators(self, indicator_list: List[str], symbol: str) -> None:
        
        """
        Calculate the requested indicators 
        and add them to the main dataframe.
        """

        for indicator in indicator_list:
            if indicator in self.indicators:
                self.indicators[indicator](symbol)

                if indicator not in self.current_indicators_names:
                    
                    self.current_indicators_names.add(
                        f"{indicator}_{symbol.split('_')[-1]}"
                        )

    def _compute_rsi(self, symbol: str, window: int = 14) -> None:
        
        """
        Calculates RSI indicator for the given symbol and adds it as a
        column to the self.dataframes dataframe
        """

        symbol_suffix = symbol.split('_')[-1]
        delta = self.dataframes[symbol].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        self.dataframes['RSI_'+symbol_suffix] = rsi

    def _compute_cci(self, symbol: str, window: int = 20) -> None:
        
        """
        Calculates RSI indicator for the given symbol and adds it as a
        column to the self.dataframes dataframe
        """

        symbol_suffix = symbol.split('_')[-1]
        TP = (self.dataframes['High_'+symbol_suffix] + 
              self.dataframes['Low_'+symbol_suffix] + 
              self.dataframes[symbol]) / 3
        rolling_mean = TP.rolling(window=window).mean()
        rolling_std = TP.rolling(window=window).std()
        self.dataframes['CCI_'+symbol_suffix] = (
                TP - rolling_mean) / (0.015 * rolling_std
                )
        
    def _compute_atr(self, symbol: str, window: int = 14) -> None:
        
        """
        Calculates ATR indicator for the given symbol and adds it as a
        column to the self.dataframes dataframe
        """

        high = self.dataframes[f"High_{symbol}"]
        low = self.dataframes[f"Low_{symbol}"]
        close = self.dataframes[symbol]
        atr = high - low
        atr = atr.where(atr > (high - close).abs(), high - close)
        atr = atr.where(atr > (low - close).abs(), low - close)
        self.dataframes[f"ATR_{symbol}"] = atr.rolling(window=window).mean()

    def _compute_ma(self, symbol: str, window: int = 14) -> None:
        
        """
        Calculates SMA indicator for the given symbol and adds it as a
        column to the self.dataframes dataframe
        """

        close = self.dataframes[symbol]
        self.dataframes[f"MA_{symbol}"] = close.rolling(window=window).mean()

    def _compute_ema(self, symbol: str, window: int = 14) -> None:
        
        """
        Calculates EMA indicator for the given symbol and adds it as a
        column to the self.dataframes dataframe
        """

        close = self.dataframes[symbol]
        self.dataframes[f"EMA_{symbol}"] = close.ewm(span=window, adjust=False).mean()

    def _compute_stochastic(self, symbol: str, window: int = 14) -> None:
        
        """
        Calculates Stochastic indicator for the given symbol and adds it as a
        column to the self.dataframes dataframe
        """

        high = self.dataframes[f"High_{symbol}"]
        low = self.dataframes[f"Low_{symbol}"]
        close = self.dataframes[symbol]
        low_min = low.rolling(window=window).min()
        high_max = high.rolling(window=window).max()
        self.dataframes[f"Stoch_{symbol}"] = 100 * (close - low_min) / (high_max - low_min)

    def _compute_bollinger_bands(self, symbol: str, window: int = 14) -> None:
        
        """
        Calculates BB indicator for the given symbol and adds it as a
        column to the self.dataframes dataframe
        """

        close = self.dataframes[symbol]
        ma = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()
        self.dataframes[f"BBUpper_{symbol}"] = ma + (std * 2)
        self.dataframes[f"BBLower_{symbol}"] = ma - (std * 2)

# --------------------------------------------------------------
# Usage Example:
symbols = ["SPY", "QQQ"]
analyzer = MarketDataModule(symbols)
analyzer.fetch_data("2000-01-01")
analyzer.calculate_indicators(['RSI'], "Close_SPY")
analyzer.calculate_indicators(['CCI'], "Close_QQQ")
analyzer.calculate_indicators(['BB'], "Close_SPY")
data = analyzer.get_data()
print(data)

plt.plot(data["BBUpper_Close_SPY"])
plt.plot(data["BBLower_Close_SPY"])
plt.show()
