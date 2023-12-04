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
        self.rsi_period = 14
        self.mfi_period = 14
        self.mfi_upper_band = 80
        self.mfi_lower_band = 20
        self.stochastic_oscillator_period = 14
        self.stochastic_oscillator_upper_band = 80
        self.stochastic_oscillator_lower_band = 20
        self.roc_period = 9
        self.bollinger_band_period = 20
        self.on_balance_volumne_period = 20

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
    
    def plot_macd_signal(self, macd, signal, macd_label_txt: str, sig_label_txt: str, title_txt: str):
        fig, ax = plt.subplots(figsize=(15, 9))
        ax.plot(macd, label = f"{macd_label_txt}", color= 'red')
        ax.plot(signal, label = f"{sig_label_txt}", color= 'blue')
        ax.set_title(f"{title_txt}", color = 'black', fontsize = 20)
        ax.set_xlabel('Date', color = 'black', fontsize = 15)
        ax.legend(loc='upper left')
        return fig
    

    def buy_sell_macd(self, signal):
        Buy = []
        Sell = []
        flag = -1

        for i in range(0, len(signal)):
            if signal['MACD'][i] < signal['Signal Line'][i]:
                Sell.append(np.nan)
                if flag != 1:
                    Buy.append(signal['Adj Close'][i])
                    flag = 1
                else:
                    Buy.append(np.nan)
            elif signal['MACD'][i] > signal['Signal Line'][i]:
                Buy.append(np.nan)
                if flag != 0:
                    Sell.append(signal['Adj Close'][i])
                    flag = 0
                else:
                    Sell.append(np.nan)
            else:
                Buy.append(np.nan)
                Sell.append(np.nan)

        return (Buy, Sell)
    
    def buy_sell_macd_plot(self, data, title_txt: str):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.scatter(data.index, data['Buy_Signal_Price'], color='green', label='Buy', marker='^', alpha=1)
        ax.scatter(data.index, data['Sell_Signal_Price'], color='red', label='Sell', marker='v', alpha=1)
        ax.plot(data['Adj Close'], label='Adj Close Price', alpha = 0.35)
        ax.set_title(f"{title_txt}", color = 'black', fontsize = 20)
        ax.set_xlabel('Date', color = 'black', fontsize = 15)
        ax.set_ylabel('Adj Close Price')
        ax.legend(loc = 'upper left')
        return fig
    
    def plot_rsi(self, title_txt: str, rsi_data):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.set_title(f"{title_txt}", color = 'black', fontsize = 20)
        ax.set_xlabel('Date', color = 'black', fontsize = 15)
        ax.set_ylabel('RSI', color = 'black', fontsize = 15)
        rsi_data.plot(ax=ax)
        return fig
    

    def plot_rsi_with_sma(self, data, title_txt: str):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.set_title(f"{title_txt}", color = 'black', fontsize = 20)
        ax.plot(data['RSI'])
        ax.set_xlabel('Date', color = 'black', fontsize = 15)
        ax.axhline(0, linestyle='--', alpha = 0.5, color='gray')
        ax.axhline(10, linestyle='--', alpha = 0.5, color='orange')
        ax.axhline(20, linestyle='--', alpha = 0.5, color='green')
        ax.axhline(30, linestyle='--', alpha = 0.5, color='red')
        ax.axhline(70, linestyle='--', alpha = 0.5, color='red')
        ax.axhline(80, linestyle='--', alpha = 0.5, color='green')
        ax.axhline(90, linestyle='--', alpha = 0.5, color='orange')
        ax.axhline(100, linestyle='--', alpha = 0.5, color='gray')
        return fig
    
    def plot_rsi_with_ewma(self, data, title_txt: str):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.set_title(f"{title_txt}", color = 'black', fontsize = 20)
        ax.set_xlabel('Date', color = 'black', fontsize = 15)
        ax.plot(data['RSI2'])
        ax.axhline(0, linestyle='--', alpha = 0.5, color='gray')
        ax.axhline(10, linestyle='--', alpha = 0.5, color='orange')
        ax.axhline(20, linestyle='--', alpha = 0.5, color='green')
        ax.axhline(30, linestyle='--', alpha = 0.5, color='red')
        ax.axhline(70, linestyle='--', alpha = 0.5, color='red')
        ax.axhline(80, linestyle='--', alpha = 0.5, color='green')
        ax.axhline(90, linestyle='--', alpha = 0.5, color='orange')
        ax.axhline(100, linestyle='--', alpha = 0.5, color='gray')
        return fig
    
    def plot_mfi(self, data, title_txt: str):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(data['MFI'], label = 'MFI')
        ax.axhline(10, linestyle = '--', color = 'orange')
        ax.axhline(20, linestyle = '--', color = 'blue')
        ax.axhline(80, linestyle = '--', color = 'blue')
        ax.axhline(90, linestyle = '--', color = 'orange')
        ax.set_title(f"{title_txt}", color = 'black', fontsize = 20)
        ax.set_xlabel('Time periods', color = 'black', fontsize = 15)
        ax.set_ylabel('MFI Values', color = 'black', fontsize = 15)
        return fig
    
    def mfi_buy_sell_plot(self, data, title_txt: str):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(data['Close'], label = 'Close Price', alpha = 0.5)
        ax.scatter(data.index, data['Buy'], color = 'green', label = 'Buy Signal', marker = '^', alpha = 1)
        ax.scatter(data.index, data['Sell'], color = 'red', label = 'Sell Signal', marker = 'v', alpha = 1)
        ax.set_title(f"{title_txt}", color = 'black', fontsize = 20)
        ax.set_xlabel('Date', color = 'black', fontsize = 15)
        ax.set_ylabel('Close Price', color = 'black', fontsize = 15)
        ax.legend(loc='upper left')
        return fig
    
    def plot_stochastic_oscillator(self, data, title_txt: str):
        fig, ax = plt.subplots(figsize=(20, 10))
        data[['Strategy Returns','Market Returns']].cumsum().plot(ax=ax)
        ax.set_title(title_txt, color = 'black', fontsize = 20)
        return fig
    
    def plot_bollinger_bands(self, data, column_list, title_txt: str):
        fig, ax = plt.subplots(figsize=(20, 10))
        data[column_list].plot(ax=ax)
        plt.style.use('ggplot')
        ax.set_title(title_txt, color = 'black', fontsize = 20)
        ax.set_ylabel('Close Price', color = 'black', fontsize = 15)
        return fig
    
    def plot_bollinger_bands_shaded(self, data, title_txt: str):
        fig, ax = plt.subplots(figsize=(20,10))
        x_axis = data.index
        ax.fill_between(x_axis, data['Upper'], data['Lower'], color='grey')
        ax.plot(data['Close'], color='gold', lw=3, label = 'Close Price') #lw = line width
        ax.plot(data['SMA'], color='blue', lw=3, label = 'Simple Moving Average')
        ax.set_title(title_txt, color = 'black', fontsize = 20)
        ax.set_xlabel('Date', color = 'black', fontsize = 15)
        ax.set_ylabel('Close Price', color = 'black', fontsize = 15)
        plt.xticks(rotation = 45)
        ax.legend()
        return fig
    
    def plot_bollinger_bands_shaded_with_signals(self, data, title_txt: str):
        fig, ax = plt.subplots(figsize=(20,10))
        x_axis = data.index
        ax.fill_between(x_axis, data['Upper'], data['Lower'], color='grey')
        ax.plot(data['Close'], color='gold', lw=3, label = 'Close Price', alpha = 0.5)
        ax.plot(data['SMA'], color='blue', lw=3, label = 'Moving Average', alpha = 0.5)
        ax.scatter(x_axis, data['Buy'], color='green', lw=3, label = 'Buy', marker = '^', alpha = 1)
        ax.scatter(x_axis, data['Sell'], color='red', lw=3, label = 'Sell', marker = 'v', alpha = 1)
        ax.set_title(title_txt, color = 'black', fontsize = 20)
        ax.set_xlabel('Date', color = 'black', fontsize = 15)
        ax.set_ylabel('Close Price', color = 'black', fontsize = 15)
        plt.xticks(rotation = 45)
        ax.legend()
        return fig
    
    def plot_obv_ema(self, data, title_txt: str):
        fig, ax = plt.subplots(figsize=(17, 8))
        plt.style.use('ggplot')
        ax.plot(data['OBV'], label = 'OBV', color = 'orange')
        ax.plot(data['OBV_EMA'], label = 'OBV_EMA', color = 'purple')
        ax.set_title(title_txt, color = 'black', fontsize = 20)
        ax.set_xlabel('Date', fontsize = 15)
        ax.set_ylabel('Price', fontsize = 15)
        ax.legend(loc = 'upper left')
        return fig
    
    def buy_sell_obv_plot(self, data, title_txt: str):
        fig, ax = plt.subplots(figsize=(17, 8))
        plt.style.use('ggplot')
        ax.plot(data['Adj Close'], label = 'Adjusted Close', alpha = 0.5)
        ax.scatter(data.index, data['Buy'], label = 'Buy Signal', marker = '^', alpha = 1, color = 'green')
        ax.scatter(data.index, data['Sell'], label = 'Sell Signal', marker = 'v', alpha = 1, color = 'red')
        ax.set_title(title_txt, color = 'black', fontsize = 20)
        ax.set_xlabel('Date', fontsize = 15)
        ax.set_ylabel('Price', fontsize = 15)
        ax.legend(loc = 'upper left')
        return fig
        

    def ui_renderer(self):
        st.title('Stonks 📈')
        st.image('https://i.ytimg.com/vi/if-2M3K1tqk/maxresdefault.jpg')

        # 
        # TODO: Update details with markdown and add more meta docs
        # 
        st.write('Welcome to Stonks, a simple web app that allows you to analyze stocks.')
        st.write("Final Year Project by: [Sayan Kumar Ghosh](gsayankr02@gmail.com), [Vishal Choubey](vishalchoubey1019@gmail.com), [Jit Karan](jitkaran55@gmail.com), [Soumili Saha](ssoumilisaha2001@gmail.com), [Shubhayu Majumdar](shubhayumajumdar64@gmail.com)")
        
        st.markdown("""---""")

        # Sidebar Inputs
        stock_names = [stock.name for stock in self.stocks]
        self.selected_stock = st.sidebar.selectbox('Select a Stock:', stock_names)
        self.selected_ticker = self.stocks[stock_names.index(self.selected_stock)].ticker
        self.start_date = st.sidebar.date_input('Start date', date.today() - timedelta(weeks=52))
        self.end_date = st.sidebar.date_input('End date', date.today())
        self.stick = st.sidebar.selectbox('Stick', ["day", "week", "month", "year"])
        st.sidebar.markdown("""---""")
        self.rsi_period = st.sidebar.number_input('RSI Period', 14, 100, 14)
        st.sidebar.markdown("""---""")
        self.mfi_period = st.sidebar.number_input('MFI Period', 14, 100, 14)
        self.mfi_upper_band = st.sidebar.number_input('MFI Upper Band', 50, 100, 80)
        self.mfi_lower_band = st.sidebar.number_input('MFI Lower Band', 0, 50, 20)
        st.sidebar.markdown("""---""")
        self.stochastic_oscillator_period = st.sidebar.number_input('Stochastic Oscillator Period', 14, 100, 14)
        self.stochastic_oscillator_upper_band = st.sidebar.number_input('Stochastic Oscillator Upper Band', 50, 100, 80)
        self.stochastic_oscillator_lower_band = st.sidebar.number_input('Stochastic Oscillator Lower Band', 0, 50, 20)
        st.sidebar.markdown("""---""")
        self.roc_period = st.sidebar.number_input('ROC Period', 9, 100, 9)
        st.sidebar.markdown("""---""")
        self.bollinger_band_period = st.sidebar.number_input('Bollinger Band Period', 20, 100, 20)
        st.sidebar.markdown("""---""")
        self.on_balance_volumne_period = st.sidebar.number_input('On Balance Volumne Period', 20, 100, 20)


        # Assertions for all inputs
        if not self.start_date <= self.end_date:
            st.error("Error: Start date must fall before end date.")
            st.toast("Error: Start date must fall before end date.")
            st.stop()
        
        if not self.end_date <= date.today():
            st.error("Error: End date must not be in the future.")
            st.toast("Error: End date must not be in the future.")
            st.stop()

        st.subheader(f"Stonks Analysis on {self.selected_stock} from {self.start_date} to {self.end_date}")

        self.get_stock_data()

        if self.stock_df.empty:
            st.error("Error: No data found for selected stock.")
            st.stop()

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

        st.markdown("""
            ### Moving Average Convergence Divergence (MACD)
                
            The MACD is a trend-following momentum indicator turning two trend-following indicators, moving averages, into a momentum oscillator by subtracting the longer moving average from the shorter one.

            It is useful although lacking one prediction element - because it is unbounded it is not particularly useful for identifying overbought and oversold levels. Traders can look for signal line crossovers, neutral/centreline crossovers (otherwise known as the 50 level) and divergences from the price action to generate signals. 

            The default parameters are 26 EMA of prices, 12 EMA of prices and a 9-moving average of the difference between the first two.
        """)

        short_ema = self.stock_df['Adj Close'].ewm(span=12, adjust=False).mean()
        long_ema = self.stock_df['Adj Close'].ewm(span=26, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=9, adjust=False).mean()

        st.pyplot(self.plot_macd_signal(macd, signal, macd_label_txt=f"{self.selected_stock} MACD", sig_label_txt=f"Signal Line", title_txt=f"MACD and Signal Line for {self.selected_stock} stock"))


        temp_df = self.stock_df.copy()
        temp_df['MACD'] = macd
        temp_df['Signal Line'] = signal
        temp_df['Buy_Signal_Price'], temp_df['Sell_Signal_Price'] = self.buy_sell_macd(temp_df)
        
        st.write("When the MACD line crosses above the signal line this indicates a good time to buy.")
        st.pyplot(self.buy_sell_macd_plot(temp_df, title_txt=f"MACD Buy and Sell Signals for {self.selected_stock} stock"))


        st.markdown("""
            ## Momentum Strategies
            
            In momentum algorithmic trading strategies stocks have momentum (i.e. upward or downward trends) that we can detect and exploit.
            
            ### Relative Strength Index (RSI)

            The RSI is a momentum indicator. A typical momentum strategy will buy stocks that have been showing an upward trend in hopes that the trend will continue, and make predictions based on whether the past recent values were going up or going down. 

            The RSI determines the level of overbought (70) and oversold (30) zones using a default lookback period of 14 i.e. it uses the last 14 values to calculate its values. The idea is to buy when the RSI touches the 30 barrier and sell when it touches the 70 barrier. 
        """)

        def get_rsi():
            temp_df = self.stock_df.copy()
            delta = temp_df['Adj Close'].diff(1)
            delta.dropna(inplace=True)
            up, down = delta.clip(lower=0), -delta.clip(upper=0)

            # Relative Strength Index
            rs = up.rolling(window=self.rsi_period).mean() / down.abs().rolling(window=self.rsi_period).mean()

            # Relative Strength Index with Expoential Weighted Moving Average
            rs_ewma = up.ewm(span=self.rsi_period).mean() / down.abs().ewm(span=self.rsi_period).mean()

            return 100 - (100 / (1 + rs)), 100 - (100 / (1 + rs_ewma))

        # PLot RSI with SMA
        temp_df = self.stock_df.copy()
        temp_df['RSI'], temp_df['RSI2'] = get_rsi()

        st.pyplot(self.plot_rsi(title_txt=f"RSI for {self.selected_stock} stock", rsi_data=temp_df["RSI"]))
        st.pyplot(self.plot_rsi_with_sma(temp_df, title_txt=f"RSI with {self.rsi_period}-day SMA for {self.selected_stock} stock"))
        st.pyplot(self.plot_rsi_with_ewma(temp_df, title_txt=f"RSI with {self.rsi_period}-day EWMA for {self.selected_stock} stock"))

        st.markdown("""
            ### Money Flow Index (MFI)
            
            Money Flow Index (MFI) is a technical oscillator, and momentum indicator, that uses price and volume data for identifying overbought or oversold signals in an asset. It can also be used to spot divergences which warn of a trend change in price. The oscillator moves between 0 and 100 and a reading of above 80 implies overbought conditions, and below 20 implies oversold conditions.

            It is related to the Relative Strength Index (RSI) but incorporates volume, whereas the RSI only considers price. 
        """)

        def get_mfi():
            temp_df = self.stock_df.copy()
            typical_price = (temp_df['High'] + temp_df['Low'] + temp_df['Close']) / 3
            money_flow = typical_price * temp_df['Volume']

            # Get all positive and negative money flows
            positive_flow = []
            negative_flow = []

            # Loop through typical price
            for i in range(1, len(typical_price)):
                if typical_price[i] > typical_price[i-1]:
                    positive_flow.append(money_flow[i-1])
                    negative_flow.append(0)
                elif typical_price[i] < typical_price[i-1]:
                    negative_flow.append(money_flow[i-1])
                    positive_flow.append(0)
                else:
                    positive_flow.append(0)
                    negative_flow.append(0)
            
            positive_mf = []
            negative_mf = []

            for i in range(self.mfi_period-1, len(positive_flow)):
                positive_mf.append(sum(positive_flow[i + 1 - self.mfi_period : i+1]))
            for i in range(self.mfi_period-1, len(negative_flow)):
                negative_mf.append(sum(negative_flow[i + 1 - self.mfi_period : i+1]))
            
            mfi = 100 * (np.array(positive_mf) / (np.array(positive_mf) + np.array(negative_mf)))
            mfi = np.append([np.nan]*self.mfi_period, mfi)

            return mfi
        
        temp_df = self.stock_df.copy()
        temp_df['MFI'] = get_mfi()
        st.pyplot(self.plot_mfi(temp_df, title_txt=f"MFI for {self.selected_stock} stock"))

        def get_mfi_signal(data, high, low):
            buy_signal = []
            sell_signal = []

            for i in range(len(data['MFI'])):
                if data['MFI'][i] > high:
                    buy_signal.append(np.nan)
                    sell_signal.append(data['Close'][i])
                elif data['MFI'][i] < low:
                    buy_signal.append(data['Close'][i])
                    sell_signal.append(np.nan)
                else:
                    sell_signal.append(np.nan)
                    buy_signal.append(np.nan)

            return (buy_signal, sell_signal)
        
        temp_df["Buy"], temp_df["Sell"] = get_mfi_signal(temp_df, self.mfi_upper_band, self.mfi_lower_band)
        st.pyplot(self.mfi_buy_sell_plot(temp_df, title_txt=f"MFI Buy and Sell Signals for {self.selected_stock} stock"))

        st.markdown("""
            ### Stochastic Oscillator

            The stochastic oscillator is a momentum indicator comparing the closing price of a security to the range of its prices over a certain period of time and is one of the best-known momentum indicators along with RSI and MACD.

            The intuition is that in a market trending upward, prices will close near the high, and in a market trending downward, prices close near the low.

            The stochastic oscillator is plotted within a range of zero and 100. The default parameters are an overbought zone of 80, an oversold zone of 20 and well-used lookbacks period of 14 and 5 which can be used simultaneously. The oscillator has two lines, the %K and %D, where the former measures momentum and the latter measures the moving average of the former. The %D line is more important of the two indicators and tends to produce better trading signals which are created when the %K crosses through the %D.
            """)
        
        st.markdown("""
            The stochastic oscillator is calculated using the following formula:
            ```python
            %K = 100(C – L)/(H – L)
            ```

            Where:

            C -> the most recent closing price

            L -> the low of the given previous trading sessions

            H -> the highest price traded during the same given period

            %K -> the current market rate for the currency pair

            %D -> 3-period moving average of %K
        """)
       

        def get_stochastic_oscillator():
            temp_df = self.stock_df.copy()
            
            temp_df["L"] = temp_df['Low'].rolling(window=self.stochastic_oscillator_period).min()
            temp_df["H"] = temp_df['High'].rolling(window=self.stochastic_oscillator_period).max()
            temp_df['%K'] = 100*((temp_df['Close'] - temp_df['L']) / (temp_df['H'] - temp_df['L']) )
            temp_df['%D'] = temp_df['%K'].rolling(window=3).mean()
            temp_df['Sell Entry'] = ((temp_df['%K'] < temp_df['%D']) & (temp_df['%K'].shift(1) > temp_df['%D'].shift(1))) & (temp_df['%D'] > self.stochastic_oscillator_upper_band) 
            temp_df['Sell Exit'] = ((temp_df['%K'] > temp_df['%D']) & (temp_df['%K'].shift(1) < temp_df['%D'].shift(1))) 

            temp_df['Short'] = np.where(temp_df['Sell Entry'], -1, np.where(temp_df['Sell Exit'], 0, 0))
            temp_df['Short'] = temp_df['Short'].fillna(method='pad')
            temp_df['Buy Entry'] = ((temp_df['%K'] > temp_df['%D']) & (temp_df['%K'].shift(1) < temp_df['%D'].shift(1))) & (temp_df['%D'] < self.stochastic_oscillator_lower_band) 
            temp_df['Buy Exit'] = ((temp_df['%K'] < temp_df['%D']) & (temp_df['%K'].shift(1) > temp_df['%D'].shift(1))) 

            temp_df['Long'] = np.where(temp_df['Buy Entry'], 1, np.where(temp_df['Buy Exit'], 0, 0))
            temp_df['Long'] = temp_df['Long'].fillna(method='pad')
            temp_df['Long'][0] = 0

            temp_df['Position'] = temp_df['Long'] + temp_df['Short']
            temp_df['Market Returns'] = temp_df['Close'].pct_change()
            temp_df['Strategy Returns'] = temp_df['Market Returns'] * temp_df['Position'].shift(1)

            return temp_df

        temp_df = get_stochastic_oscillator()
        st.pyplot(self.plot_stochastic_oscillator(temp_df, title_txt=f"Stochastic Oscillator for {self.selected_stock} stock"))

        st.markdown("""
            ###  Rate of Change (ROC) 

            The ROC indicator is a pure momentum oscillator. The ROC calculation compares the current price with the price "n" periods ago e.g. when we compute the ROC of the daily price with a 9-day lag, we are simply looking at how much, in percentage, the price has gone up (or down) compared to 9 days ago. Like other momentum indicators, ROC has overbought and oversold zones that may be adjusted according to market conditions. 
        """)
        
        def get_roc():
            temp_df =  self.stock_df.copy()
            temp_df['ROC'] = ((temp_df['Adj Close'] - temp_df['Adj Close'].shift(self.roc_period)) / temp_df['Adj Close'].shift(self.roc_period) -1 ) * 100

            return temp_df
        
        temp_df = get_roc()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(mpf.plot(temp_df, type='candle',  style='yahoo', figsize=(15,8),  title=f"{self.selected_stock} Daily Price", volume=True))

        st.markdown("""
            ## Volatility trading strategies

            Volatility trading involves predicting the stability of an asset’s value. Instead of trading on the price rising or falling, traders take a position on whether it will move in any direction. Volatility trading is a good option for investors who believe that the price of an asset will stay within a certain range.

            ### Bollinger Bands
                    
            A Bollinger Band is a volatility indicator based on based on the correlation between the normal distribution and stock price and can be used to draw support and resistance curves. It is defined by a set of lines plotted two standard deviations (positively and negatively) away from a simple moving average (SMA) of the security's price, but can be adjusted to user preferences.

            By default it calculates a 20-period SMA (the middle band), an upper band two standard deviations above the the moving average and a lower band two standard deviations below it.

            If the price moves above the upper band this could indicate a good time to sell, and if it moves below the lower band it could be a good time to buy. 

            Whereas the RSI can only be used as a confirming factor inside a ranging market, not a trending market, by using Bollinger bands we can calculate the widening variable, or moving spread between the upper and the lower bands, that tells us if prices are about to trend and whether the RSI signals might not be that reliable.

            Despite 90% of the price action happening between the bands, however, a breakout is not necessarily a trading signal as it provides no clue as to the direction and extent of future price movement.
        """)


        def get_bollinger_bands():
            temp_df = self.stock_df.copy()
            temp_df['SMA'] = temp_df['Close'].rolling(window=self.bollinger_band_period).mean()
            temp_df['STD'] = temp_df['Close'].rolling(window=self.bollinger_band_period).std()
            temp_df['Upper'] = temp_df['SMA'] + (temp_df['STD'] * 2)
            temp_df['Lower'] = temp_df['SMA'] - (temp_df['STD'] * 2)
            column_list = ['Close', 'SMA', 'Upper', 'Lower']

            return temp_df, column_list
        

        temp_df, column_list = get_bollinger_bands()
        st.pyplot(self.plot_bollinger_bands(temp_df, column_list, title_txt=f"Bollinger Bands for {self.selected_stock} stock"))
        st.pyplot(self.plot_bollinger_bands_shaded(temp_df, title_txt=f"Shaded Bollinger Bands region for {self.selected_stock} stock"))
        
        def get_signal_bb(data):
            buy_signal = [] 
            sell_signal = [] 

            for i in range(len(data['Close'])):
                if data['Close'][i] > data['Upper'][i]: 
                    buy_signal.append(np.nan)
                    sell_signal.append(data['Close'][i])
                elif data['Close'][i] < data['Lower'][i]:
                    sell_signal.append(np.nan)
                    buy_signal.append(data['Close'][i])
                else:
                    buy_signal.append(np.nan)
                    sell_signal.append(np.nan)
            return (buy_signal, sell_signal)
        
        temp_df['Buy'], temp_df['Sell'] = get_signal_bb(temp_df)
        st.pyplot(self.plot_bollinger_bands_shaded_with_signals(temp_df, title_txt=f"Bollinger Bands with Buy and Sell Signals for {self.selected_stock} stock"))
        

        st.markdown("""
            ## Volume Trading Strategies

            Volume trading is a measure of how much of a given financial asset has traded in a period of time. Volume traders look for instances of increased buying or selling orders. They also pay attention to current price trends and potential price movements. Generally, increased trading volume will lean heavily towards buy orders.        
            
            ### On Balance Volume (OBV)

            OBV is a momentum-based indicator which measures volume flow to gauge the direction of the trend. Volume and price rise are directly proportional and OBV can be used as a confirmation tool with regards to price trends. A rising price is depicted by a rising OBV and a falling OBV stands for a falling price. 

            It is a  cumulative total of the up and down volume. When the close is higher than the previous close, the volume is added to the running
            total, and when the close is lower than the previous close, the volume is subtracted
            from the running total. 
        """)

        def get_obv():
            OBV = []
            OBV.append(0)

            for i in range(1, len(temp_df['Adj Close'])):
                if temp_df['Adj Close'][i] > temp_df['Adj Close'][i-1]:
                    OBV.append(OBV[-1] + temp_df.Volume[i])
                elif temp_df['Adj Close'][i] < temp_df['Adj Close'][i-1]:
                    OBV.append(OBV[-1] - temp_df.Volume[i])
                else:
                    OBV.append(OBV[-1])
            return OBV
        
        temp_df = self.stock_df.copy()
        temp_df['OBV'] = get_obv()
        temp_df['OBV_EMA'] = temp_df['OBV'].ewm(span=self.on_balance_volumne_period).mean()

        st.pyplot(self.plot_obv_ema(temp_df, title_txt=f"On Balance Volume for {self.selected_stock} stock"))

        def buy_sell_obv(signal, col1, col2):
            sigPriceBuy = []
            sigPriceSell = []
            flag = -1

            for i in range(0, len(signal)):
                # If OBV > OBV_EMA then buy --> col1 => If OBV < OBV_EMA then sell => 'OBV_EMA'
                if signal[col1][i] < signal[col2][i] and flag != 1:
                    sigPriceBuy.append(signal['Adj Close'][i])
                    sigPriceSell.append(np.nan)
                    flag = 1
                # If OBV < OBV_EMA then sell
                elif signal[col1][i] > signal[col2][i] and flag != 0:
                    sigPriceSell.append(signal['Adj Close'][i])
                    sigPriceBuy.append(np.nan)
                    flag = 0
                else:
                    sigPriceSell.append(np.nan)
                    sigPriceBuy.append(np.nan)

            return (sigPriceBuy, sigPriceSell)

        temp_df['Buy'], temp_df['Sell'] = buy_sell_obv(temp_df, 'OBV', 'OBV_EMA')

        st.pyplot(self.buy_sell_obv_plot(temp_df, title_txt=f"On Balance Volume Buy and Sell Signals for {self.selected_stock} stock"))
        
        st.markdown("""---""")
        st.markdown("""
            ## Conclusion

            It is almost certainly better to choose technical indicators that complement each other, not just those that move in unison and generate the same signals. The intuition here is that the more indicators you have that confirm each other, the better your chances are to profit. This can be done by combining strategies to form a system, and looking for multiple signals.                    
        """)

stonks = Stonks(stocks_filepath="Models/stocknames.csv")
stonks.ui_renderer()