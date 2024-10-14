import pandas as pd
from sqlalchemy import create_engine
#import ta_indicators.ta as ta
import os
from datetime import datetime as dt
import traceback

class combine_pvol_funda():
    '''
    Requires pricevol_funda.datapipe.py to have been run first. 
    This class further processes the data to prepare it for feature engineering. 
    
        assign_quarter_labels(pricevol):
            Helper function. Assigns quarter labels by organizing data into 4 bins. 
        
        assign_funda_labels(funda):
            Helper function. Organizes the fundamental data pull into 4 bins corresponding to a quarter each. 
        
        merge_pvol_funda(target_stocks: list)
            Main Function
            Ingest a list of target stocks, if the labels dont exist add them to each. Then save them as combined csv.
            Ready for ingesting into FeatureEngineering Pipeline or further processing

    Usage Example
        # Define the database connection string
        db_connection_string = 'postgresql://eqty:1234@192.168.17.4:5432/FinancialData'

        # Define the schema name
        schema_name = 'comb_pvol_funda'
        
        etl = dps.combine_pvol_funda(db_connection_string, schema_name)
        etl.download_data(target_stocks, start_date, end_date)
        for stock in target_stocks:
            stock_data = etl.extract_data(stock)
            print(stock_data.head())
    '''
    def __init__(self, schema_name='comb_pvol_funda'):
        ''' Initializes the class with the schema name and database connection string'''
        # Fetch credentials from environment variables
        pguser = os.getenv('pguser')
        pgpass = os.getenv('pgpass')
        pghost = os.getenv('pghost')

        self.db_connection_string = f'postgresql://{pguser}:{pgpass}@{pghost}/FinancialData'
        self.schema_name = schema_name
        self.engine = create_engine(self.db_connection_string)
    
    def assign_quarter_labels(self, pricevol: pd.DataFrame):
        ''' Helper function to assign quarter labels to the price-vol data '''
        # Ensure the 'date' column is in datetime format
        #pricevol['date'] = pd.to_datetime(pricevol['date'],'%Y-%m-%d')
        
        # Extract year and quarter
        pricevol['Year'] = pricevol['date'].dt.year
        pricevol['Quarter'] = pd.cut(pricevol['date'].dt.month, bins=[0, 3, 6, 9, 12], labels=['Q1', 'Q2', 'Q3', 'Q4'])
        
        # Create Quarter_Label
        pricevol['Quarter_Label'] = pricevol['Quarter'].astype(str) + pricevol['Year'].astype(str)
        
        return pricevol

    def assign_funda_labels(self, funda: pd.DataFrame):
        # Ensure the 'period_ending' column is in datetime format
        # If period ending exists, then make sure its date_time
        try:
            funda['period_ending'] = pd.to_datetime(funda['period_ending'])
        except:
            pass
        # Extract year and quarter
        funda['Year'] = funda['period_ending'].dt.year
        funda['Quarter'] = pd.cut(funda['period_ending'].dt.month, bins=[0, 3, 6, 9, 12], labels=['Q1', 'Q2', 'Q3', 'Q4'])
        
        # Create the Quarter_Label
        funda['Quarter_Label'] = funda['Quarter'].astype(str) + funda['Year'].astype(str)
        
        return funda

    def merge_pvol_funda(self, target_stocks: list):
        ''' Take a list of target stocks, assign quarter labels for price-vol and fundamental table value matching and fill forwards.
            Designed to help prep the data for feature engineering
            Returns clean_df which is the pvol fundamental data
            
            Requires: 
            IBKR Gateway running on HAL @ 192.168.17.5
            ibkr_pvol.py
            fundamental.py
            ta_indicators package for calculation
            
        '''
        for stock in target_stocks:
            try:
                try: 
                    # Download pricevol data from databento schema, stock symbol table
                    pricevol = pd.read_sql_table(stock, self.engine, schema='databento_ohlcv')
                    # Databento specific renaming
                    pricevol['date'] = pd.to_datetime(pricevol['ts_event'],'%Y-%m-%d')
                    print(f'PriceVol data loaded for {stock}')
                    # Download fundamental data from fundamental schema, stock symbol table
                    funda = pd.read_sql_table(stock, self.engine, schema='fundamental')
                    print(f'Fundamental data loaded for {stock}')
                except:
                    print(f"PriceVol or Fundamental table for {stock} is missing. Run the datapipe first!")
                    break
                            
                pricevol = self.assign_quarter_labels(pricevol)
                funda = self.assign_funda_labels(funda)
                ### Combiner logic: Combine the pricevol and fundamental tables ###
                
                # Merge on Quarter_Label
                merged_df = pd.merge(pricevol, funda, on='Quarter_Label', how='left')
                
                # Drop duplicate or unnecessary columns
                try:
                    # lift the rest of this from ML features, since that is the most up to date version, find the code to drop anything with _y in it too
                    clean_df = merged_df.drop(columns=['symbol_y', 'period_ending', 'Year_y','Quarter_y','ts_event','rtype', 'publisher_id','instrument_id'])
                except KeyError:
                    print(f"There was an error joining the tables for {stock}, 'symbol_y' and 'period_ending' are missing")
                    traceback.print_exc()
                    continue
                
                try:
                    # Rename the columns
                    clean_df = clean_df.rename(columns={'symbol_x': 'symbol','Year_x': 'year', 'Quarter_x': 'quarter'})
                except KeyError:
                    print(f"There was an error joining the tables for {stock}, 'symbol_y' and 'period_ending' are missing")
                    traceback.print_exc()
                    continue

                # Write the data to SQL
                table_name = f"{stock}"
                clean_df.to_sql(table_name, self.engine, if_exists='replace', index=False, schema='comb_pvol_funda')
                print(f"Price-volume fundamental data for {stock} saved to table {self.schema_name}.{table_name}")
                
            except Exception as e:
                print(f"Error for {stock}. Skipping...")
                traceback.print_exc()
                pass
            #return clean_df
    
    def get_comb_data_from_postgresql(self, ticker, start_date=None, end_date=None, schema='comb_pvol_funda', date_column='date'):
        """
        Retrieves data for a given ticker from PostgreSQL database, optionally within a date range.
        """
        try:
            # Start building the query
            query = f'SELECT * FROM "{schema}"."{ticker}"'
            
            # Add date range filtering if both start_date and end_date are provided
            if start_date is not None and end_date is not None:
                # Convert date strings to datetime objects
                start_date = dt.strptime(start_date, '%Y-%m-%d')
                end_date = dt.strptime(end_date, '%Y-%m-%d')
                
                # Append the date filtering to the query, cast the timestamp to a date
                query += f" WHERE CAST({date_column} AS DATE) BETWEEN '{start_date.strftime('%Y-%m-%d')}' AND '{end_date.strftime('%Y-%m-%d')}'"
            
            # Execute the query and read the data into a DataFrame
            df = pd.read_sql(query, con=self.engine)
            
            return df
        
        except Exception as e:
            print(f"Error retrieving data for {ticker} from PostgreSQL: {e}")
            return None