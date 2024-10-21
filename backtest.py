import xgboost as xgb
import pandas as pd
import databento_sqlmod1 as dbs
#import databento_sql as dbs
import fmp_keymetrics
import combiner
import numpy as np
#import talib as ta  # ta-lib or ta for calculating technical indicators
import ta
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import joblib
from datetime import datetime

''' Notes
Look at the trade_log and come up with something to tally the trade_log to find the portfolio value
Then add in portfolio allocations (equal weighting across tickers)

Need to redo this project. Portfolio logging isnt working correctly. Convert it to quantconnect
'''

ticker_list = ['AAPL', 'MSFT', 'NVDA',  # Technology
    'AMZN', 'TSLA', 'HD',  # Consumer Discretionary
    'UNH', 'JNJ', 'LLY',  # Healthcare
    'JPM', 'BAC', 'WFC',  # Financials
    'XOM', 'CVX', 'COP',  # Energy
    'PG', 'KO', 'PEP',  # Consumer Staples
    'BA', 'CAT', 'UPS',  # Industrials
    'LIN', 'APD', 'SHW',  # Materials
    'PLD', 'AMT', 'CCI',  # Real Estate
    'NEE', 'DUK', 'SO',  # Utilities
]

# Databento download
for ticker in ticker_list:
    dbs.download_and_append_data(ticker, '2019-01-01', '2023-12-31', frequency='daily')

#FMP Download
start_date = '2019-01-01'
end_date = '2024-09-31'

fmp = fmp_keymetrics.funda_ETL()
keymetrics = fmp.download_funda_data(ticker_list, start=start_date, end=end_date)
keymetrics

# Merge Databento and FMP data
comb = combiner.combine_pvol_funda()
#tickers = ['AAPL','AMZN','GOOGL','MSFT']
merged = comb.merge_pvol_funda(ticker_list)

# Initialize combiner instance
comb = combiner.combine_pvol_funda()

def add_technical_indicators(df):
    # Add SMA, EMA
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema_100'] = ta.trend.ema_indicator(df['close'], window=100)
    df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
    
    # Add RSI, MACD
    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd(df['close'])
    df['macd_signal'] = ta.trend.macd_signal(df['close'])

    # Add Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'], window=20)
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_lower'] = bollinger.bollinger_lband()
    
    # Add Average True Range (ATR)
    df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    
    # Add On-Balance Volume (OBV)
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    
    # Add Stochastic Oscillator (K, D)
    stoch = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch
    df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    
    # Remove momentum since it was causing an error
    # Add Williams %R
    df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
    
    # Add Chaikin Money Flow (CMF)
    df['cmf'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], window=20)
    
    return df

def prepdata(ticker, start_date, end_date):
    # Retrieve data for the ticker and date range
    data = comb.get_comb_data_from_postgresql(ticker, start_date, end_date)

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Sort the data by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    # Define the fundamental columns and technical indicators to use as features
    fundamental_columns = [
        'revenuePerShare', 'netIncomePerShare', 'operatingCashFlowPerShare', 'freeCashFlowPerShare', 
        'cashPerShare', 'bookValuePerShare', 'tangibleBookValuePerShare', 'shareholdersEquityPerShare', 
        'interestDebtPerShare', 'marketCap', 'enterpriseValue', 'peRatio', 'priceToSalesRatio', 
        'pocfratio', 'pfcfRatio', 'pbRatio', 'ptbRatio', 'evToSales', 'enterpriseValueOverEBITDA', 
        'evToOperatingCashFlow', 'evToFreeCashFlow', 'earningsYield', 'freeCashFlowYield', 
        'debtToEquity', 'debtToAssets', 'netDebtToEBITDA', 'currentRatio', 'interestCoverage', 
        'incomeQuality', 'dividendYield', 'payoutRatio', 'salesGeneralAndAdministrativeToRevenue', 
        'researchAndDdevelopementToRevenue', 'intangiblesToTotalAssets', 'capexToOperatingCashFlow', 
        'capexToRevenue', 'capexToDepreciation', 'stockBasedCompensationToRevenue', 'grahamNumber', 
        'roic', 'returnOnTangibleAssets', 'grahamNetNet', 'workingCapital', 'tangibleAssetValue', 
        'netCurrentAssetValue', 'investedCapital', 'averageReceivables', 'averagePayables', 
        'averageInventory', 'daysSalesOutstanding', 'daysPayablesOutstanding', 
        'daysOfInventoryOnHand', 'receivablesTurnover', 'payablesTurnover', 'inventoryTurnover', 
        'roe', 'capexPerShare'
    ]

    # Technical columns added
    technical_columns = [
        'sma_20', 'sma_50', 'ema_20', 'rsi_14', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 
        'atr_14', 'obv', 'stoch_k', 'stoch_d', 'williams_r', 'cmf', 'ema_50', 'ema_100', 'ema_200'
    ]

    # Add technical indicators
    df = add_technical_indicators(df)

    # Calculate log returns as the target variable
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Fill missing values in the fundamental columns using forward fill, followed by backward fill, then zeros if necessary
    df[fundamental_columns + technical_columns] = df[fundamental_columns + technical_columns].ffill().bfill().fillna(0)

    # Drop rows with NaN values in log returns and technical indicators
    df = df.dropna()

    # Define features (X) and target (y)
    X = df[fundamental_columns + technical_columns]
    y = df['log_return']

    # Convert features into an XGBoost DMatrix
    dmat = xgb.DMatrix(X, label=y, missing=np.nan)

    return dmat, y, X

# Example of how to use the updated function
#ticker_list = ['AAPL']
for ticker in tqdm(ticker_list, desc="Processing tickers", unit="ticker"):
    start_date = '2019-01-01'
    end_date = '2022-12-31'

    # Prepare the training data using log returns
    dtrain, y_train, X_train = prepdata(ticker, start_date, end_date)

    ''' Setting Test Data '''
    start_date = '2023-01-01'
    end_date = '2023-12-31'

    # Prepare the test data
    dtest, y_test, X_test = prepdata(ticker, start_date, end_date)
    

def log_trade(trade_log, ticker, date, position_type, trade_type, entry_price, exit_price=None, num_shares=None, signal=None, portfolio_value_start=None, portfolio_value_end=None):
    """
    Log trade details into the trade_log DataFrame.
    """
    return_on_trade = None
    if exit_price is not None and entry_price is not None:
        return_on_trade = (exit_price - entry_price) / entry_price

    trade_entry = pd.DataFrame({
        'ticker': [ticker],
        'date': [date],
        'position_type': [position_type],  # 'long' or 'short'
        'trade_type': [trade_type],  # 'buy' or 'sell'
        'entry_price': [entry_price],
        'exit_price': [exit_price],
        'return': [return_on_trade],
        'num_shares': [num_shares],  # Track the number of shares bought/sold
        'signal': [signal],
        'portfolio_value_start': [portfolio_value_start],  # Log portfolio value at start of trade
        'portfolio_value_end': [portfolio_value_end]  # Log portfolio value at end of trade
    })

    trade_log = pd.concat([trade_log, trade_entry], ignore_index=True)
    
    return trade_log

def calculate_portfolio_metrics(trade_log_df):
    """
    Calculate portfolio metrics based on the trade log.
    """
    if trade_log_df['return'].empty or trade_log_df['return'].isna().all():
        return {'Sharpe Ratio': float('nan'), 'Sortino Ratio': float('nan'), 'Max Drawdown': float('nan')}
    
    trade_returns = trade_log_df['return'].dropna()
    sharpe_ratio = (trade_returns.mean() / trade_returns.std()) * np.sqrt(252) if trade_returns.std() > 0 else float('nan')
    downside_risk = trade_returns[trade_returns < 0].std()
    sortino_ratio = (trade_returns.mean() / downside_risk) * np.sqrt(252) if downside_risk > 0 else float('nan')

    cumulative_return = (1 + trade_returns).cumprod()
    drawdown = cumulative_return / cumulative_return.cummax() - 1
    max_drawdown = drawdown.min()

    return {
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown
    }

# Load the trained model
best_model = joblib.load('best_xgb_model.pkl')

# Initialize variables for backtesting
initial_cash = 100000  # Initial capital
cash = initial_cash
portfolio = {ticker: [] for ticker in ticker_list}  # Track multiple positions (shares and entry prices) in each stock

# Initialize DataFrames
trade_log = pd.DataFrame(columns=['ticker', 'date', 'position_type', 'trade_type', 'entry_price', 'exit_price', 'return', 'signal'])
portfolio_value_df = pd.DataFrame(columns=['date', 'portfolio_value'])

# Load your price data for 2022 and 2023
def get_data(ticker, start_date, end_date):
    """
    Fetch data from PostgreSQL; if missing, download from Databento.
    """
    df = dbs.get_data_from_postgresql(ticker, start_date, end_date)

    if df is None or df.empty:
        print(f"Data for {ticker} not found in PostgreSQL. Fetching from Databento.")
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        df = dbs.get_data_from_databento(ticker, start_date, end_date)

        dbs.upload_to_postgresql(df, ticker)

    df['date'] = pd.to_datetime(df['ts_event'])
    df = df.sort_values(by='date')

    return df

# Backtesting loop for 2023
for ticker in ticker_list:
    train_data = get_data(ticker, '2019-01-01', '2022-12-31')
    test_data = get_data(ticker, '2023-01-01', '2023-12-31')

    dtrain, y_train, X_train = prepdata(ticker, '2019-01-01', '2022-12-31')
    dtest, y_test, X_test = prepdata(ticker, '2023-01-01', '2023-12-31')

    # Make sure all test data is aligned
    print(f"Test data shape: {test_data.shape}, X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    y_pred = best_model.predict(X_test)

    # Track portfolio value daily
    for i in range(len(y_pred) - 1):
        if i + 1 >= len(test_data):
            break

        next_open = test_data['open'].iloc[i + 1]
        predicted_change = y_pred[i]
        signal = 'long' if predicted_change > 0 else 'short'

        # Buy signal
        if predicted_change > 0.01:
            num_shares = cash // next_open  # Integer division for full shares
            if num_shares > 0:
                portfolio[ticker].append({'shares': num_shares, 'entry_price': next_open})
                cash -= num_shares * next_open

                # Log buy trade
                trade_log = log_trade(
                    trade_log=trade_log,
                    ticker=ticker,
                    date=test_data['date'].iloc[i + 1],
                    position_type='long',
                    trade_type='buy',
                    entry_price=next_open,
                    num_shares=num_shares,
                    signal=signal,
                    portfolio_value_start=cash + sum([p['shares'] * next_open for p in portfolio[ticker]])
                )

        # Sell signal
        elif predicted_change < -0.01 and len(portfolio[ticker]) > 0:
            exit_price = next_open
            total_return = 0

            # Sell all shares and calculate return for each laddered entry
            for position in portfolio[ticker]:
                entry_price = position['entry_price']
                shares = position['shares']
                trade_return = (exit_price - entry_price) / entry_price
                total_return += trade_return * shares

            # Log sell trade
            total_shares = sum([p['shares'] for p in portfolio[ticker]])
            cash += total_shares * exit_price
            portfolio[ticker] = []  # Clear out all positions after selling

            trade_log = log_trade(
                trade_log=trade_log,
                ticker=ticker,
                date=test_data['date'].iloc[i + 1],
                position_type='long',
                trade_type='sell',
                entry_price=None,
                exit_price=exit_price,
                num_shares=total_shares,
                signal=signal,
                portfolio_value_end=cash
            )

        # Calculate portfolio value daily and append to rolling DataFrame
        portfolio_value = cash + sum([p['shares'] * test_data['close'].iloc[i + 1] for ticker, positions in portfolio.items() for p in positions])
        portfolio_value_df = pd.concat([portfolio_value_df, pd.DataFrame({'date': [test_data['date'].iloc[i + 1]], 'portfolio_value': [portfolio_value]})], ignore_index=True)

# Calculate final portfolio value
final_portfolio_value = cash + sum([sum([p['shares'] for p in portfolio[ticker]]) * test_data['close'].iloc[-1] for ticker in ticker_list])

print(f"Final Portfolio Value: {final_portfolio_value}")

# Save portfolio value log to CSV
portfolio_value_df.to_csv('portfolio_value_log.csv', index=False)

# Calculate portfolio performance metrics
portfolio_metrics = calculate_portfolio_metrics(trade_log)
print(portfolio_metrics)

# Update Sharpe, Sortino, and Max Drawdown calculations to use the new rolling portfolio value.
def calculate_portfolio_metrics(trade_log_df, portfolio_value_df):
    # Calculate daily returns from portfolio value changes
    portfolio_value_df['daily_return'] = portfolio_value_df['portfolio_value'].pct_change().fillna(0)

    sharpe_ratio = (portfolio_value_df['daily_return'].mean() / portfolio_value_df['daily_return'].std()) * np.sqrt(252)
    
    downside_risk = portfolio_value_df[portfolio_value_df['daily_return'] < 0]['daily_return'].std()
    sortino_ratio = (portfolio_value_df['daily_return'].mean() / downside_risk) * np.sqrt(252) if downside_risk > 0 else float('nan')

    # Max drawdown calculation
    cumulative_return = (1 + portfolio_value_df['daily_return']).cumprod()
    drawdown = cumulative_return / cumulative_return.cummax() - 1
    max_drawdown = drawdown.min()

    return {
        'Sharpe Ratio': sharpe_ratio if portfolio_value_df['daily_return'].std() > 0 else float('nan'),
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown
    }

# Recalculate portfolio metrics using the rolling portfolio value
portfolio_metrics = calculate_portfolio_metrics(trade_log, portfolio_value_df)
print(portfolio_metrics)
trade_log.to_csv('trade_log.csv', index=False)
