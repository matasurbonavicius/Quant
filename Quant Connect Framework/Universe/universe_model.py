from AlgorithmImports import *
from datetime import timedelta


class UniverseModel:
    
    def __init__(self, algorithm: QCAlgorithm) -> None:

        """
        Initialize the UniverseSelectionModel.

        Args:
            algo: The algorithm instance.
        """

        self.algo = algorithm


    def signal_and_tradable_instrument_model(self, 
        signal_instrument: object,
        tradable_instrument: object,
        filter: dict = {"From": 0, "To": 180}
        ) -> None:

        """
        Notes:
            Signal_instrument is always the benchmark
            Signal instrument is used for calculation
            Tradable instrument is used to trade
            Example: SPX for signals, ES Future to trade

        Args:
            * signal_instrument: QC Securities Object. The benchmark
            * tradable_instrument: QC Securities Object
            * filter: If any symbol is a future, pass an opt. dictionary for the filter
        

        Global access:
            instruments are saved in QCAlgorithm class:
            * self.algo.signal_instrument
            * self.algo.signal_instrument
            * self.algo.universe_symbols - array of all symbols
        
        Compatible with:
            * one_instrument_model
        """

        self.algo.signal_instrument = signal_instrument
        self.algo.tradable_instrument = tradable_instrument
        self.algo.universe_symbols = []

        instruments_without_expiry = [
            SecurityType.Equity, 
            SecurityType.Index
            ]

        # Type of the instrument
        signal_type = signal_instrument.Symbol.SecurityType
        tradable_type = tradable_instrument.Symbol.SecurityType

        # If any of the instruments are futures, add a filter
        if signal_type == SecurityType.Future:
            signal_instrument.SetFilter(
                timedelta(days=filter["From"]),
                timedelta(days=filter["To"])
                )
        elif signal_type in instruments_without_expiry:
            self.algo.universe_symbols.append(signal_instrument.Symbol)
        else:
            self.algo.Error(
                "UniverseModel: Signal type not supported, adjust code"
                )

        if tradable_type == SecurityType.Future:
            tradable_instrument.SetFilter(
                timedelta(days=filter["From"]),
                timedelta(days=filter["To"])
                )
        elif tradable_type in instruments_without_expiry:
            self.algo.universe_symbols.append(tradable_instrument.Symbol)
        else:
            self.algo.Error(
                "UniverseModel: Tradable type not supported, adjust code"
                )
        
        # Check if at least one instrument has been added
        if len(self.algo.universe_symbols) == 0:
            self.algo.Error(
                "UniverseSelectionModel: One instrument should be Index or Equity"
                )
        
        # Use Quant Connects default Manual Universe Selection Model -
        # - to add our symbols
        self.algo.SetUniverseSelection(
            ManualUniverseSelectionModel(self.algo.universe_symbols)
            )

        # Set benchmark
        self.algo.SetBenchmark(signal_instrument.Symbol)

    
    def one_instrument_model(self, 
        instrument: object,
        filter: dict = {"From": 0, "To": 180}
        ) -> None:
        
        """
         Notes:
            Use only one instrument for algorithm
            Both signal and tradable instruments are the same

        Args:
            * instrument: QC Securities Object. The benchmark
            * filter: If any symbol is a future, pass an opt. dictionary for the filter

        Global access:
            instruments are saved in QCAlgorithm class:
            * self.algo.signal_instrument
            * self.algo.tradable_instrument

        Compatible with:
            * signal_and_tradable_instrument_model
        """

        self.algo.signal_instrument = instrument
        self.algo.tradable_instrument = instrument
        self.algo.universe_symbols = []

        instrument_type = instrument.Symbol.SecurityType

        # If the instrument is future, add a filter
        if instrument_type == SecurityType.Future:
            instrument.SetFilter(
                timedelta(days=filter["From"]),
                timedelta(days=filter["To"])
                )
        
        self.algo.universe_symbols.append(instrument.Symbol)

        # Use Quant Connects default Manual Universe Selection Model -
        # - to add our symbols
        self.algo.SetUniverseSelection(
            ManualUniverseSelectionModel(self.algo.universe_symbols)
            )

        # Set benchmark
        self.algo.SetBenchmark(instrument.Symbol)
