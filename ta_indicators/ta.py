import pandas as pd
import ta_indicators.momentum as ta_mom
import ta_indicators.moving_averages as ta_ma
import ta_indicators.volume as ta_volume
import ta_indicators.volatility as ta_vol

class ta_adder():
    def addcol(self, df: pd.DataFrame):
        ''' Adds Technical Analysis indicators from the custom library to the input dataframe'''
        
        # Add RSI
        df['RSI'] = ta_mom.relative_strength_index(df['Adj Close'], period=14)
        
        # Add EMAs
        df['Fast EMA'] = ta_ma.exponential_moving_average(df['Adj Close'], period=10)
        df['Slow EMA'] = ta_ma.exponential_moving_average(df['Adj Close'], period=20)
        
        # Add OBV
        df['OBV'] = ta_volume.on_balance_volume(df['Adj Close'], df['Volume'])
        
        # Calculate MACD and append it to the DataFrame
        macd_line, signal_line, macd_histogram = ta_ma.macd(df['Adj Close'], fast_period=12, slow_period=26, signal_period=9)
        #df['MACD_Line'] = macd_line #consider dropping this for feature engineering
        #df['Signal_Line'] = signal_line #consider dropping this for feature engineering
        df['MACD_Histogram'] = macd_histogram # this is the important one for feature engineering
        
        # Add Bollinger Bads
        df['Bollinger_Upper_Band'], df['Bollinger_Lower_Band'] = ta_vol.bollinger_bands(df['Adj Close'], period=20, std_dev=2)
        
        return df