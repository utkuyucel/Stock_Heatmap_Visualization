import yfinance as yf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
from matplotlib.colors import LinearSegmentedColormap


class StockPerformanceAnalyzer:
    def __init__(self, stock_symbol, start_date):
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.performance_data = self._get_stock_performance_transposed()

    def _get_stock_performance_transposed(self):
        stock_data = yf.download(self.stock_symbol, start=self.start_date, end=date.today())
        stock_data['Month'] = stock_data.index.to_period('M')
        monthly_performance = stock_data.groupby('Month').apply(
            lambda x: (x['Adj Close'][-1] - x['Adj Close'][0]) / x['Adj Close'][0]
        )
        monthly_performance = monthly_performance.reset_index()
        monthly_performance['Year'] = monthly_performance['Month'].dt.year
        monthly_performance['Month'] = monthly_performance['Month'].dt.month
        performance_pivot = monthly_performance.pivot("Year", "Month", 0)
        
        return performance_pivot

    def plot_heatmap(self):
        colored_data_with_nan = self._color_data_with_nan()
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        xticks = [month_names[month] for month in self.performance_data.columns]
        
        plt.figure(figsize=(17, 8))
        sns.heatmap(
            colored_data_with_nan, 
            annot=self.performance_data, 
            fmt=".2%", 
            cmap=self._get_colormap(), 
            cbar_kws={
                'label': 'Monthly Performance',
                'ticks': [-1, 0, 1, 2], 
                'boundaries': [-1.5, -0.5, 0.5, 1.5, 2.5]
            }, 
            linewidths=0.5, 
            xticklabels=xticks, 
            yticklabels=self.performance_data.index
        )
        plt.title(f'Monthly Performance of {self.stock_symbol} from {self.start_date} to Today')
        plt.show()
        plt.subplots_adjust(hspace=0.8)

    def plot_monthly_histogram(self):
        self._plot_histogram(axis=0)

    def plot_yearly_histogram(self):
        self._plot_histogram(axis=1)

    def _plot_histogram(self, axis):
        positive_counts, negative_counts = self._get_positive_negative_counts(axis)
        xlabel = "Month" if axis == 0 else "Year"
        
        index = np.arange(len(positive_counts))
        bar_width = 0.35
        
        plt.figure(figsize=(10, 5))
        bar1 = plt.bar(index, positive_counts, bar_width, label='Positive', color='green')
        bar2 = plt.bar(index + bar_width, negative_counts, bar_width, label='Negative', color='red')

        # Add data values on top of the bars
        for bars in (bar1, bar2):
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height, '%d' % int(height), ha='center', va='bottom')

        plt.xlabel(xlabel)
        plt.ylabel('Counts')
        plt.title(f'Counts of Positive and Negative Performance by {xlabel}')
        plt.xticks(index + bar_width/2, positive_counts.index)  
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.subplots_adjust(hspace=0.8)

    def _get_positive_negative_counts(self, axis):
        positive_counts = (self.performance_data > 0).sum(axis=axis)
        negative_counts = (self.performance_data <= 0).sum(axis=axis)
        return positive_counts, negative_counts

    def _color_data_with_nan(self):
        return np.vectorize(self._apply_color_with_nan)(self.performance_data)

    def _apply_color_with_nan(self, value):
        if np.isnan(value):
            return -1
        elif value > 0.1:
            return 2
        elif 0 <= value <= 0.1:
            return 1
        else:
            return 0

    def _get_colormap(self):
        colors_with_nan = ["grey", "red", "yellow", "green"]
        return LinearSegmentedColormap.from_list("", colors_with_nan)


if __name__ == "__main__":
    analyzer = StockPerformanceAnalyzer("TUPRS.IS", "2010-01-01")
    analyzer.plot_heatmap()
    analyzer.plot_monthly_histogram()
    analyzer.plot_yearly_histogram()
