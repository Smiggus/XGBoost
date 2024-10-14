import pandas as pd

def simple_moving_average(data, period=20):
    return data.rolling(window=period).mean()

def exponential_moving_average(data, period=20):
    return data.ewm(span=period, adjust=False).mean()

def macd(data, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = exponential_moving_average(data, fast_period)
    slow_ema = exponential_moving_average(data, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = exponential_moving_average(macd_line, signal_period)
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram