from typing import List, Optional
import yfinance as yf
from datetime import date, timedelta
import streamlit as st
import pandas as pd
import numpy as np
from Models.datamodels import StockNameModel
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.dates import date2num, DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY
import seaborn as sns
import mplfinance as mpf
from mplfinance.original_flavor import candlestick_ohlc



class Stonks:
    def __init__(self, stocks_filepath: str) -> None:
        # Classwise global variables
        self.stocks = None
        self.selected_stock = None
        self.selected_ticker = None
        self.start_date = date.today() - timedelta(weeks=52)
        self.end_date = date.today()
        self.stock_df = None
        self.stick = "day"

        # Init functions
        self.stocksFilePath = stocks_filepath
        self.get_stocks()

    def get_stocks(self):
        if self.stocks is None:
            stocksNames = pd.read_csv(self.stocksFilePath)
            self.stocks = [StockNameModel(name=row['name'], ticker=row['ticker']) for index, row in stocksNames.iterrows()]

    def get_stock_data(self):
        self.stock_df = yf.download(self.selected_ticker, self.start_date, self.end_date)

    def pandas_candlestick_ohlc(self, dat, txt, stick = "day", otherseries = None) :
        """
        Japanese candlestick chart showing OHLC prices for a specified time period
        
        :param dat: pandas dataframe object with datetime64 index, and float columns "Open", "High", "Low", and "Close"
        :param stick: A string or number indicating the period of time covered by a single candlestick. Valid string inputs include "day", "week", "month", and "year", ("day" default), and any numeric input indicates the number of trading days included in a period
        :param otherseries: An iterable that will be coerced into a list, containing the columns of dat that hold other series to be plotted as lines
    
        :returns: a Japanese candlestick plot for stock data stored in dat, also plotting other series if passed.
        """
        mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
        alldays = DayLocator()              # minor ticks on the days
        dayFormatter = DateFormatter('%d')      # e.g., 12
    
        # Create a new DataFrame which includes OHLC data for each period specified by stick input
        transdat = dat.loc[:,["Open", "High", "Low", "Close"]]
        if (type(stick) == str):
            if stick == "day":
                plotdat = transdat
                stick = 1 # Used for plotting
            elif stick in ["week", "month", "year"]:
                if stick == "week":
                    transdat["week"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[1]) # Identify weeks
                elif stick == "month":
                    transdat["month"] = pd.to_datetime(transdat.index).map(lambda x: x.month) # Identify months
                transdat["year"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[0]) # Identify years
                grouped = transdat.groupby(list(set(["year",stick]))) # Group by year and other appropriate variable
                plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted
                for name, group in grouped:
                    temporary_entry = pd.DataFrame({"Open": group.iloc[0,0],
                                            "High": max(group.High),
                                            "Low": min(group.Low),
                                            "Close": group.iloc[-1,3]},
                                            index = [group.index[0]],)
                    temporary_entry.name = group.index[0]
                    plotdat = pd.concat([plotdat, temporary_entry], ignore_index = False)
                if stick == "week": stick = 5
                elif stick == "month": stick = 30
                elif stick == "year": stick = 365
        elif (type(stick) == int and stick >= 1):
            transdat["stick"] = [np.floor(i / stick) for i in range(len(transdat.index))]
            grouped = transdat.groupby("stick")
            plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted
            for name, group in grouped:
                temporary_entry = pd.DataFrame({"Open": group.iloc[0,0],
                                            "High": max(group.High),
                                            "Low": min(group.Low),
                                            "Close": group.iloc[-1,3]},
                                            index = [group.index[0]],)
                temporary_entry.name = group.index[0]
                plotdat = pd.concat([plotdat, temporary_entry], ignore_index = False)
    
        else:
            raise ValueError('Valid inputs to argument "stick" include the strings "day", "week", "month", "year", or a positive integer')
    
        # print(plotdat)
        # Set plot parameters, including the axis object ax used for plotting
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2)
        """
        if plotdat.index[-1] - plotdat.index[0] < pd.Timedelta('730 days'):
            weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
            ax.xaxis.set_major_locator(mondays)
            ax.xaxis.set_minor_locator(alldays)
        """
        if pd.Timedelta(f"{plotdat.index[-1] - plotdat.index[0]} days") < pd.Timedelta('730 days'):
            weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
            ax.xaxis.set_major_locator(mondays)
            ax.xaxis.set_minor_locator(alldays)
        else:
            weekFormatter = DateFormatter('%b %d, %Y')
        ax.xaxis.set_major_formatter(weekFormatter)
    
        ax.grid(True)
    
        # Create the candelstick chart
        candlestick_ohlc(ax, list(zip(list(date2num(plotdat.index.tolist())), plotdat["Open"].tolist(), plotdat["High"].tolist(),
                        plotdat["Low"].tolist(), plotdat["Close"].tolist())),
                        colorup = "green", colordown = "red", width = stick * .4)
    
        # Plot other series (such as moving averages) as lines
        if otherseries != None:
            if type(otherseries) != list:
                otherseries = [otherseries]
            dat.loc[:,otherseries].plot(ax = ax, lw = 1.3, grid = True)
    
        ax.xaxis_date()
        ax.autoscale_view()

        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
        sns.set(rc={'figure.figsize':(20, 10)})
        plt.style.use('ggplot')
        plt.title(f"Candlestick chart of {txt}", color = 'black', fontsize = 20)
        plt.xlabel('Date', color = 'black', fontsize = 15)
        plt.ylabel('Stock Price (p)', color = 'black', fontsize = 15)
    
        # plt.show()
        return fig
    
    def sma(self, title_txt: str, label_txt: str, days: List[int]):
        fig, ax = plt.subplots(figsize=(15,9))
        for day in days:
            self.stock_df['Adj Close'].loc[str(self.start_date):str(self.end_date)].rolling(window=day).mean().plot(ax=ax, label=f'{day} Day Avg')
        self.stock_df['Adj Close'].loc[str(self.start_date):str(self.end_date)].plot(ax=ax, label=f"{label_txt}")
        ax.set_title(f"{title_txt}", color = 'black', fontsize = 20)
        ax.set_xlabel('Date', color = 'black', fontsize = 15)
        ax.set_ylabel('Stock Price (p)', color = 'black', fontsize = 15)
        ax.legend()
        return fig
    
    def ewma(self, title_txt: str, label_txt: str, days: List[int]):
        fig, ax = plt.subplots(figsize=(15,9))
        for day in days:
            self.stock_df['Adj Close'].loc[str(self.start_date):str(self.end_date)].ewm(window=day).mean().plot(ax=ax, label=f'{day} Day Avg')
        self.stock_df['Adj Close'].loc[str(self.start_date):str(self.end_date)].plot(ax=ax, label=f"{label_txt}")
        ax.set_title(f"{title_txt}", color = 'black', fontsize = 20)
        ax.set_xlabel('Date', color = 'black', fontsize = 15)
        ax.set_ylabel('Stock Price (p)', color = 'black', fontsize = 15)
        ax.legend()
        return fig
    
    def tripple_ewma(self, title_txt: str, label_txt: str, short_ema_span: int, middle_ema_span: int, long_ema_span: int):
        fig, ax = plt.subplots(figsize=(15,9))

        ax.plot(self.stock_df['Adj Close'].loc[str(self.start_date):str(self.end_date)], label=f"{label_txt}", color = 'blue')
        ax.plot(self.stock_df.loc[str(self.start_date):str(self.end_date)]['Adj Close'].ewm(span=short_ema_span, adjust=False).mean(), label = 'Short/Fast EMA', color = 'red')
        ax.plot(self.stock_df.loc[str(self.start_date):str(self.end_date)]['Adj Close'].ewm(span=middle_ema_span, adjust=False).mean(), label = 'Middle/Medium EMA', color = 'orange')
        ax.plot(self.stock_df.loc[str(self.start_date):str(self.end_date)]['Adj Close'].ewm(span=long_ema_span, adjust=False).mean(), label = 'Long/Slow EMA', color = 'green')

        ax.set_title(f"{title_txt}", color = 'black', fontsize = 20)
        ax.set_xlabel('Date', color = 'black', fontsize = 15)
        ax.set_ylabel('Stock Price (p)', color = 'black', fontsize = 15)
        ax.legend()
        return fig
    
    def buy_sell_triple_ewma(self, data):
        buy_list = []
        sell_list = []
        flag_long = False
        flag_short = False

        for i in range(0, len(data)):
            if data['Middle'][i] < data['Long'][i] and data['Short'][i] < data['Middle'][i] and flag_long == False and flag_short == False:
                buy_list.append(data['Adj Close'][i])
                sell_list.append(np.nan)
                flag_short = True
            elif flag_short == True and data['Short'][i] > data['Middle'][i]:
                sell_list.append(data['Adj Close'][i])
                buy_list.append(np.nan)
                flag_short = False
            elif data['Middle'][i] > data['Long'][i] and data['Short'][i] > data['Middle'][i] and flag_long == False and flag_short == False:
                buy_list.append(data['Adj Close'][i])
                sell_list.append(np.nan)
                flag_long = True
            elif flag_long == True and data['Short'][i] < data['Middle'][i]:
                sell_list.append(data['Adj Close'][i])
                buy_list.append(np.nan)
                flag_long = False
            else:
                buy_list.append(np.nan)
                sell_list.append(np.nan)
        
        return (buy_list, sell_list)
    
    def buy_sell_ewma3_plot(self, data, label_txt: str, title_txt: str):
        fig, ax = plt.subplots(figsize=(18, 10))

        ax.plot(data['Adj Close'], label=f"{label_txt}", color = 'blue', alpha = 0.35)
        ax.plot(data["Short"], label = 'Short/Fast EMA', color = 'red', alpha = 0.35)
        ax.plot(data["Middle"], label = 'Middle/Medium EMA', color = 'orange', alpha = 0.35)
        ax.plot(data["Long"], label = 'Long/Slow EMA', color = 'green', alpha = 0.35)
        ax.scatter(data.index, data['Buy'], color = 'green', label = 'Buy Signal', marker = '^', alpha = 1)
        ax.scatter(data.index, data['Sell'], color = 'red', label = 'Sell Signal', marker='v', alpha = 1)

        ax.set_title(f"{title_txt}", color = 'black', fontsize = 20)
        ax.set_xlabel('Date', color = 'black', fontsize = 15)
        ax.set_ylabel('Stock Price (p)', color = 'black', fontsize = 15)
        ax.legend()
        return fig
    
    def exponential_smoothing(self, series, alpha):
        result = [series[0]] # first value is same as series
        for n in range(1, len(series)):
            result.append(alpha * series[n] + (1 - alpha) * result[n-1])
        return result

    def plot_exponential_smoothing(self, series, alphas, label_txt: str, title_txt: str):
        fig, ax = plt.subplots(figsize=(17, 8))
        for alpha in alphas:
            ax.plot(self.exponential_smoothing(series, alpha), label=f"Alpha {alpha}")
        ax.plot(series.values, "c", label = f"{label_txt}")
        ax.set_xlabel('Days', color = 'black', fontsize = 15)
        ax.set_ylabel('Stock Price (p)', color = 'black', fontsize = 15)
        ax.legend(loc="best")
        ax.axis('tight')
        ax.set_title(f"{title_txt}", color = 'black', fontsize = 20)
        ax.grid(True)
        return fig
    
    def double_exponential_smoothing(self, series, alpha, beta):
        result = [series[0]]
        for n in range(1, len(series)+1):
            if n == 1:
                level, trend = series[0], series[1] - series[0]
            if n >= len(series): # forecasting
                value = result[-1]
            else:
                value = series[n]
            last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
            result.append(level + trend)
        return result

    def plot_double_exponential_smoothing(self, series, alphas, betas, label_txt: str, title_txt: str):
        fig, ax = plt.subplots(figsize=(17, 8))
        for alpha in alphas:
            for beta in betas:
                ax.plot(self.double_exponential_smoothing(series, alpha, beta), label=f"Alpha {alpha}, beta {beta}")
        ax.plot(series.values, label = f"{label_txt}")
        ax.set_xlabel('Days', color = 'black', fontsize = 15)
        ax.set_ylabel('Stock Price (p)', color = 'black', fontsize = 15)
        ax.legend(loc="best")
        ax.axis('tight')
        ax.set_title(f"{title_txt}", color = 'black', fontsize = 20)
        ax.grid(True)
        return fig
        

    def ui_renderer(self):
        st.title('Stonks📈')
        st.image('https://i.ytimg.com/vi/if-2M3K1tqk/maxresdefault.jpg')

        # Update details with markdown and add more meta docs
        st.write('Welcome to Stonks, a simple web app that allows you to analyze stocks.')
        st.write("Final Year Project by: [Shubhayu Majumdar](), [Sayan Kumar Ghosh](), [Vishal Choubey](), [Soumili Saha](), [Jit Karan](),")


        # Sidebar Inputs
        stock_names = [stock.name for stock in self.stocks]
        self.selected_stock = st.sidebar.selectbox('Select a stock:', stock_names)
        self.selected_ticker = self.stocks[stock_names.index(self.selected_stock)].ticker
        self.start_date = st.sidebar.date_input('Start date', date.today() - timedelta(weeks=52))
        self.end_date = st.sidebar.date_input('End date', date.today())
        self.stick = st.sidebar.selectbox('Stick', ["day", "week", "month", "year"])

        # Assertions for all inputs
        assert self.start_date <= self.end_date, 'Error: End date must fall after start date.'
        assert self.end_date <= date.today(), 'Error: End date must not be in the future.'

        st.subheader(f"Stonks Analysis on {self.selected_stock} from {self.start_date} to {self.end_date}")

        self.get_stock_data()

        st.dataframe(self.stock_df)

        st.header("Visualising stock data")
        st.markdown("""
                    **Japanese candlestick charts** are tools used in a particular trading style called price action to predict market movement through pattern recognition of continuations, breakouts and reversals. 
                    
                    Unlike a line chart, all of the price information can be viewed in one figure showing the high, low, open and close price of the day or chosen time frame. Price action traders observe patterns formed by green bullish candles where the stock is trending upwards over time, and red or black bearish candles where there is a downward trend.
                """)

        txt = f"{self.selected_stock} OHLC stock prices from {self.start_date} - {self.end_date}"
        st.pyplot(self.pandas_candlestick_ohlc(self.stock_df, stick=self.stick, txt = txt))

        st.header("Trend-following strategies")
        st.write("Trend-following is about profiting from the prevailing trend through  buying an asset when its price trend goes up, and selling when its trend goes down, expecting price movements to continue.")
        
        st.subheader("Moving averages")
        st.write("Moving averages smooth a series filtering out noise to help identify trends, one of the fundamental principles of technical analysis being that prices move in trends. Types of moving averages include simple, exponential, smoothed, linear-weighted, MACD, and as lagging indicators they follow the price action and are commonly referred to as trend-following indicators.")
        
        st.subheader("Simple Moving Average (SMA)")
        st.markdown("""
            The simplest form of a moving average, known as a Simple Moving Average (SMA), is calculated by taking the arithmetic mean of a given set of values over a set time period.  This model is probably the most naive approach to time series modelling and simply states that the next observation is the mean of all past observations and each value in the time period carries equal weight. 

            Modelling this an as average calculation problem we would try to predict the future stock market prices (for example, x<sub>t</sub>+1 ) as an average of the previously observed stock market prices within a fixed size window (for example, x<sub>t</sub>-n, ..., x<sub>t</sub>). This helps smooth out the price data by creating a constantly updated average price so that the impacts of random, short-term fluctuations on
            the price of a stock over a specified time-frame are mitigated.
        """, unsafe_allow_html = True)

        st.pyplot(self.sma(title_txt=f"20-day Simple Moving Average for {self.selected_stock} stock", label_txt=f"{self.selected_stock}", days=[20]))
        st.pyplot(self.sma(title_txt=f"20, 50, 100 and 200 day moving averages for {self.selected_stock} stock", label_txt=f"{self.selected_stock}", days=[20, 50, 100, 200]))
        
        st.markdown("""
                The chart shows that the 20-day moving average is the most sensitive to local changes, and the 200-day moving average the least. Here, the 200-day moving average indicates an overall bullish trend - the stock is trending upward over time. The 20- and 50-day moving averages are at times bearish and at other times bullish.

                The major drawback of moving averages, however, is that because they are lagging, and smooth out prices, they tend to recognise reversals too late and are therefore not very helpful when used alone.
        """)

        st.markdown("""
            ### Trading Strategy

            The moving average crossover trading strategy will be to take two moving averages - 20-day (fast) and 200-day (slow) - and to go long (buy) when the fast MA goes above the slow MA and to go short (sell) when the fast MA goes below the slow MA.
        """)


        temp_df = self.stock_df.copy()
        temp_df["20d"] = np.round(temp_df["Adj Close"].rolling(window = 20, center = False).mean(), 2)
        temp_df["50d"] = np.round(temp_df["Adj Close"].rolling(window = 50, center = False).mean(), 2)
        temp_df["200d"] = np.round(temp_df["Adj Close"].rolling(window = 200, center = False).mean(), 2)

        st.pyplot(self.pandas_candlestick_ohlc(temp_df.loc[str(self.start_date):str(self.end_date),:], otherseries = ["20d", "50d", "200d"], txt = txt, stick=self.stick))

        st.markdown("""
            ### Exponential Moving Average

            In a Simple Moving Average, each value in the time period carries
            equal weight, and values outside of the time period are not included in the average. However, the Exponential Moving Average is a cumulative calculation where a different decreasing weight is assigned to each observation. Past values have a diminishing contribution to the average, while more recent values have a greater contribution. This method allows the moving average to be more responsive to changes in the data.
        """)

        st.pyplot(self.sma(title_txt=f"20-day Exponential Moving Average for {self.selected_stock} stock", label_txt=f"{self.selected_stock}", days=[20]))
        st.pyplot(self.sma(title_txt=f"20, 50, 100 and 200-day Exponential Moving Averages for {self.selected_stock} stock", label_txt=f"{self.selected_stock}", days=[20, 50, 100, 200]))
        

        st.markdown("""
            ### Triple Moving Average Crossover Strategy

            This strategy uses three moving moving averages - short/fast, middle/medium and long/slow - and has two buy and sell signals. 

            The first is to buy when the middle/medium moving average crosses above the long/slow moving average and the short/fast moving average crosses above the middle/medium moving average. If we use this buy signal the strategy is to sell if the short/fast moving average crosses below the middle/medium moving average.

            The second is to buy when the middle/medium moving average crosses below the long/slow moving average and the short/fast moving average crosses below the middle/medium moving average. If we use this buy signal the strategy is to sell if the short/fast moving average crosses above the middle/medium moving average.        
        """)
        st.pyplot(self.tripple_ewma(title_txt=f"Triple Exponential Moving Average for {self.selected_stock} stock", label_txt=f"{self.selected_stock}", short_ema_span=5, middle_ema_span=21, long_ema_span=63))


        temp_df = self.stock_df.copy()
        temp_df["Short"] = temp_df["Adj Close"].ewm(span=5, adjust=False).mean()
        temp_df["Middle"] = temp_df["Adj Close"].ewm(span=21, adjust=False).mean()
        temp_df["Long"] = temp_df["Adj Close"].ewm(span=63, adjust=False).mean()

        temp_df["Buy"], temp_df["Sell"] = self.buy_sell_triple_ewma(temp_df)

        st.pyplot(self.buy_sell_ewma3_plot(temp_df, label_txt=f"{self.selected_stock}", title_txt=f"Trading signals for {self.selected_stock} stock"))


        st.markdown("""
            ### Exponential Smoothing

            Single Exponential Smoothing, also known as Simple Exponential Smoothing, is a time series forecasting method for univariate data without a trend or seasonality. It requires an alpha parameter, also called the smoothing factor or smoothing coefficient, to control the rate at which the influence of the observations at prior time steps decay exponentially.
        """)
        st.pyplot(self.plot_exponential_smoothing(self.stock_df["Adj Close"], [0.3, 0.05], label_txt=f"{self.selected_stock}", title_txt=f"Single Exponential Smoothing for {self.selected_stock} stock using 0.05 and 0.3 as alpha values"))

        st.markdown("""
            The smaller the smoothing factor (coefficient), the smoother the time series will be. As the smoothing factor approaches 0, we approach the moving average model so the smoothing factor of 0.05 produces a smoother time series than 0.3. This indicates slow learning (past observations have a large influence on forecasts). A value close to 1 indicates fast learning (that is, only the most recent values influence the forecasts).        
            
            **Double Exponential Smoothing (Holt’s Linear Trend Model)** is an extension being a recursive use of Exponential Smoothing twice where beta is the trend smoothing factor, and takes values between 0 and 1. It explicitly adds support for trends.        
            """)
        st.pyplot(self.plot_double_exponential_smoothing(self.stock_df["Adj Close"], alphas=[0.9, 0.02], betas=[0.9, 0.02], label_txt=f"{self.selected_stock}", title_txt=f"Double Exponential Smoothing for {self.selected_stock} stock with different alpha and beta values"))
        
        st.markdown("""
                The third main type is Triple Exponential Smoothing (Holt Winters Method) which is an extension of Exponential Smoothing that explicitly adds support for seasonality, or periodic fluctuations.
        """)



if __name__ == '__main__':
    stonks = Stonks(stocks_filepath="Models/stocknames.csv")
    stonks.ui_renderer()