from AlgorithmImports import *


class Charting():
    def __init__(self, algorithm, name) -> None:
        self.algo = algorithm
        
        self.algo.series = Chart(name)
        self.triangle = ScatterMarkerSymbol.Triangle
        self.circle = ScatterMarkerSymbol.Circle

        self.add_series()

    def add_series(self) -> None:

        # Adding formatted series
        self.algo.series.AddSeries(
            Series("Buy", SeriesType.Scatter, "$", Color.Green, self.circle)
            )
        self.algo.series.AddSeries(
            Series("TP", SeriesType.Scatter, "$", Color.Green, self.triangle)
            )
        self.algo.series.AddSeries(
            Series("Sell", SeriesType.Scatter, "$", Color.Red, self.triangle)
            )
        self.algo.series.AddSeries(
            Series("SL", SeriesType.Scatter, "$", Color.Black, self.triangle)
            )
        
        self.algo.AddChart(self.algo.series)
