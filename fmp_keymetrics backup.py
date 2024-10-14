import yfinance as yf
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, inspect
from sqlalchemy.sql import text

""" Check if the data is already in the database, if not download it and save it to the database """

class funda_ETL():
    def __init__(self):
        os.makedirs('./data/fundamental', exist_ok=True)
        # FinModelPrep API key
        self.fmp_api_key = os.getenv("FMP_api_key")
        
        self.db_connection_string = 'postgresql://eqty:1234@192.168.17.4:5432/FinancialData'
        self.schema_name = 'fundamental'
        self.engine = create_engine(self.db_connection_string)
        
    def download_funda_data(self, target_stocks: list, start, end) -> pd.DataFrame:
        ### Key Metrics Data ###
        key_metrics_data_list = []
        limit = 23 # This is the number of quarters to fetch
        
        # This is why I needed to extract the data from the files, can actually get rid of it and make this more efficient
        for stock in target_stocks:
            # Check if data already exists in the database
            if self.table_exists(stock) and self.data_exists(stock, start, end):
                print(f"Data for {stock} already exists in the database for the requested range.")
                continue
            # Fetch key metrics data
            url_key_metrics = f"https://financialmodelingprep.com/api/v3/key-metrics/{stock}?period=quarter&limit={limit}&apikey={self.fmp_api_key}"
            response_key_metrics = requests.get(url_key_metrics)
            data_key_metrics = response_key_metrics.json()
            print(f"Key Metrics for {stock}:")
            print(data_key_metrics)

            if len(data_key_metrics) > 0:
                df_key_metrics = pd.DataFrame(data_key_metrics)
                df_key_metrics['symbol'] = stock
                df_key_metrics = df_key_metrics.rename(columns={"date": "period_ending"}) #dont need as this is covered in combiner, better to store as period ending
                key_metrics_data_list.append(df_key_metrics)
                # Direct writes to the sql table
                table_name = f"{stock}"
                df_key_metrics.to_sql(table_name, self.engine, if_exists='replace', index=False, schema=self.schema_name)
                print(f"Fundamental data for {stock} saved to table {self.schema_name}.{table_name}")

        # This combines all the keymetrics data into a single dataframe
        key_metrics_data = pd.concat(key_metrics_data_list)
        key_metrics_data.reset_index(drop=False, inplace=False)
        return key_metrics_data

    # Helper Methods
    def table_exists(self, stock):
        """Check if table exists for the stock in the database schema."""
        inspector = inspect(self.engine)
        return inspector.has_table(stock, schema=self.schema_name)
    
    def data_exists(self, stock, start_date, end_date):
        """Check if the required data range exists in the database."""
        query = text(f"""
        SELECT COUNT(*)
        FROM "{self.schema_name}"."{stock}"
        WHERE period_ending BETWEEN :start_date AND :end_date
        """)
        
        # Create a connection and execute the query
        with self.engine.connect() as connection:
            result = connection.execute(query, {'start_date': start_date, 'end_date': end_date}).scalar()
        
        return result > 0
# Usage Example
if __name__ == '__main__':
    load_dotenv()
    etl = funda_ETL()
    target_stocks = ['AAPL']
    start_date = '2019-01-01'
    end_date = '2021-01-01'
    etl.download_funda_data(target_stocks, start_date, end_date)
    for stock in target_stocks:
        stock_data = etl.extract_data(stock)
        print(stock_data.head())