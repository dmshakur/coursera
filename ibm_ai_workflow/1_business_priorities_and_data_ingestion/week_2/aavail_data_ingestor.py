import sqlite3
import datetime
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description = 'AAVAIL Data Ingestor')
parser.add_argument('-c', '--cpath', type = String, metavar = '', required = False, help = 'path for the database')
parser.add_argument('-d', '--dpath', type = String, metavar = '', required = False, help = 'path for the df_stream')
parser.add_argument('-s', '--sql', type = String, metavar = '', required = False, help = 'query for the database')
parser.add_argument('-a', '--auto', type = bool, metavar = '', required = False, help = 'flag to automatically call the pipeline')

# group = parser.add_mutually_exclusive_group()
# group.add_argument('-q', '--quiet', action = 'store_true', help = 'print quiet')
# group.add_argument('-v', '--verbose', action = 'store_true', help = 'print verbose')

args = parser.parse_args()

class Aavail_Data_Ingestor():
    def __init__(self, connection_path = None, df_stream_path = None, pd_query = None, automate = False):
        self.connection = None
        self.customers_original = None
        self.customers = None
        self.df_streams_original = None
        self.df_streams = None
        self.customers_processed = None
        self.df_streams_processed = None
        
        if pd_query != None and connection_path != None:
            self.connect_db(connection_path)
            self.set_customers(pd_query)
        elif pd_query == None and connection_path != None:
            self.connect_db(connection_path)
        
        if df_stream_path != None:
            self.set_df_streams(df_stream_path)
        
        if automate:
#             assert self.df_streams != None or self.customers != None,\
#                 "in order to automate the process you need a proper, \
#                 connection_path, df_stream_path, and pd_query"
            self.begin_clean_pipeline()
    # Setters
    def connect_db(self, file_path):
        try:
            self.connection = sqlite3.connect(file_path)
            print("...successfully connected to db\n")
        except Error as e:
            print("...unsuccessful connection\n",e)
            
    def set_customers(self, pd_query):
        self.customers_original = pd.read_sql_query(pd_query, self.connection)
        self.customers = self.customers_original.copy()
        
    def set_df_streams(self, file_path):
        self.df_streams_original = pd.read_csv(file_path)
        self.df_streams = pd.read_csv(file_path)
    # Resetter
    def wipe_data(self, original = False, working_data = False, processed = False):
        if original:
            self.customers_original
            self.df_streams_original
        if working_data:
            self.customers
            self.df_streams
        if processed:
            self.customers_processed
            self.df_streams_processed
        
    # The pipeline - - - - - - - - - -
    def set_is_sub(self, cu, st):
        def sub_apply(row):
            return st.loc[st['customer_id'] == row['customer_id']]['subscription_stopped'].sum()
        cu['is_sub'] = cu.apply(sub_apply, axis = 1)
        return cu

    def drop_duplicate_rows(self, cu):
        return cu.drop_duplicates()

    def drop_na(self, st):
        return st.dropna(how = 'any', axis = 0)

    def set_name(self, cu):
        cu['name'] = cu['first_name'] + ' ' + cu['last_name']
        return cu.drop(['first_name', 'last_name'], axis = 1)

    def set_age(self, cu):
        def age_apply(row):
            dob = datetime.datetime.strptime(row['DOB'], '%m/%d/%y').date()
            today = datetime.datetime.today
            diff = datetime.date.today() - dob
            return int(diff.days / 365)
        cu['age'] = cu.apply(age_apply, axis = 1)
        return cu.drop(['DOB'], axis = 1)

    def set_num_streams(self, cu, st):
        def num_streams_apply(row):
            stream_count = st.loc[st['customer_id'] == row['customer_id']]
            return len(stream_count)
        cu['num_streams'] = cu.apply(num_streams_apply, axis = 1)
        return cu

    def set_sub_type(self, cu, st):
        def sub_type_apply(row):
            sub_type = st.loc[st['customer_id'] == row['customer_id']]
            return int(sub_type['invoice_item_id'].iloc[-1])
        cu['sub_type'] = cu.apply(sub_type_apply, axis = 1)
        return cu
    # End the pipeline - - - - - - - -
    
    def begin_clean_pipeline(self):
#         assert self.df_streams == None or self.customers == None, \
#             "unable to begin pipeline, customers and/or df_streams are uninitialized"
        cu = self.customers.copy()
        st = self.df_streams.copy()
        # The pipeline
        cu = self.set_is_sub(cu, st)
        print('set_is_sub, complete...')
        cu = self.drop_duplicate_rows(cu)
        print('drop_duplicate_rows, complete...')
        st = self.drop_na(st)
        print('drop_na, complete...')
        cu = self.set_name(cu)
        print('set_name, complete...')
        cu = self.set_age(cu)
        print('set_age, complete...')
        cu = self.set_num_streams(cu, st)
        print('set_num_streams, complete...')
        cu = self.set_sub_type(cu, st)
        print('set_sub_type, complete...')
        # End the pipeline
        self.customers_processed = cu
        self.df_streams_processed = st
        self.wipe_data(original = True, working_data = True)
        print('pipline, complete. returning')
        return cu, st
    
if __name__ == '__main__':
    s, c Aavail_Data_Ingestor(args.cpath, args.dpath, args.query, args.auto)
#     if args.quiet:
#         print(s, c)