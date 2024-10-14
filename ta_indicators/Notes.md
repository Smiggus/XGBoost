# Technical Analysis indicator calculations

Downloads existing data from the other feeds (directly referenced)
Uses the price-volume data to calculate custom indicators
Writes them to the table: ta_indicators on SQL

Although, might just be better to have a library that will calculate rolling windows instead. Just creating my own repo for calculations.

Currently Implemented:
    - SMA
    - EMA
    - MACD (Histogram only, represents the difference from signal line and main line)
    - RSI
    - Bollinger Bands for Volatility
    - On Balance Volume