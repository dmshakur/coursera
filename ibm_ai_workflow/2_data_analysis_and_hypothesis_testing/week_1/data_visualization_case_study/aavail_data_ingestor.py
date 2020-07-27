import sqlite3
import datetime
import pandas as pd
# import argparse

# parser = argparse.ArgumentParser(description = 'AAVAIL Data Ingestor')
# parser.add_argument('-c', '--cpath', type = String, metavar = '', required = False, help = 'path for the database')
# parser.add_argument('-d', '--dpath', type = String, metavar = '', required = False, help = 'path for the df_stream')
# parser.add_argument('-s', '--sql', type = String, metavar = '', required = False, help = 'query for the database')
# parser.add_argument('-a', '--auto', type = bool, metavar = '', required = False, help = 'flag to automatically call the pipeline')

# group = parser.add_mutually_exclusive_group()
# group.add_argument('-q', '--quiet', action = 'store_true', help = 'print quiet')
# group.add_argument('-v', '--verbose', action = 'store_true', help = 'print verbose')

# args = parser.parse_args()


class Aavail_Data_Ingestor:
    def __init__(self, connection_path=None, df_stream_path=None, pd_query=None, automate=False, verbose=False):
        self.connection = None
        self.customers_original = None
        self.customers = None
        self.df_streams_original = None
        self.df_streams = None
        self.customers_processed = None
        self.df_streams_processed = None
        self.verbose = verbose

        if pd_query != None and connection_path != None:
            self.connect_db(connection_path)
            self.set_customers(pd_query)
        elif pd_query == None and connection_path != None:
            self.connect_db(connection_path)

        if df_stream_path != None:
            self.set_df_streams(df_stream_path)

        if automate:
            self.begin_cleaning_pipeline()

    # The pipeline
    
    def begin_cleaning_pipeline(self, verbose= False):
        cu = self.customers.copy()
        st = self.df_streams.copy()
        if verbose:
            print('pipeline starting...')
        # The pipeline
        cu = self.drop_duplicate_rows(cu, verbose)
        st = self.drop_na(st, verbose)
        cu = self.set_is_sub(cu, st, verbose)
        cu = self.set_name(cu, verbose)
        cu = self.set_age(cu, verbose)
        cu = self.set_num_streams(cu, st, verbose)
        cu = self.set_sub_type(cu, st, verbose)
        cu = self.set_length_of_activity(cu, verbose)
        cu = self.impute_state(cu, verbose)
        # End the pipeline
        if verbose:
            print('pipeline complete...')

        self.customers_processed = cu
        self.df_streams_processed = st
        self.wipe_data(original=True, working_data=True)
        print('original data wiped. working_data wiped.\nprocessed data saved.\nready for new data.')
        return cu, st

    # Setters

    def connect_db(self, file_path, print_statement= False):
        try:
            self.connection = sqlite3.connect(file_path)
            if print_statement:
                print("...successfully connected to db\n")
        except:
            if print_statement:
                print("...unsuccessful connection\n")

    def set_customers(self, pd_query, print_statement= False):
        self.customers_original = pd.read_sql_query(pd_query, self.connection)
        self.customers = self.customers_original.copy()
        if print_statement:
            print('customer data set.')

    def set_df_streams(self, file_path, print_statement= False):
        self.df_streams_original = pd.read_csv(file_path)
        self.df_streams = pd.read_csv(file_path)
        if print_statement:
            print('stream data set.')

    # Resetter
    
    def wipe_data(self, original=False, working_data=False, processed=False, print_statement= False):
        if original:
            self.customers_original = None
            self.df_streams_original = None
            if print_statement:
                print('original data wiped')
        if working_data:
            self.customers = None
            self.df_streams = None
            if print_statement:
                print('working data data wiped')
        if processed:
            self.customers_processed = None
            self.df_streams_processed = None
            if print_statement:
                print('processed data wiped')

    # The pipeline - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def drop_duplicate_rows(self, cu, print_statement= False):
        if print_statement:
            print('drop duplicate rows, starting...')
        return cu.drop_duplicates()

    def drop_na(self, st, print_statement= False):
        if print_statement:
            print('drop na, starting...')
        return st.dropna(how='any', axis=0)

    def set_is_sub(self, cu, st, print_statement= False):
        if print_statement:
            print('set is sub, starting...')
        def sub_apply(row):
            return st.loc[st['customer_id'] == row['customer_id']]['subscription_stopped'].sum()
        cu['is_sub'] = cu.apply(sub_apply, axis=1)
        return cu

    def set_name(self, cu, print_statement= False):
        if print_statement:
            print('set name, starting...')
        cu['name'] = cu['first_name'] + ' ' + cu['last_name']
        return cu.drop(['first_name', 'last_name'], axis=1)

    def set_age(self, cu, print_statement= False):
        if print_statement:
            print('set age, starting...')

        def age_apply(row):
            dob = datetime.datetime.strptime(row['DOB'], '%m/%d/%y').date()
            diff = datetime.date.today() - dob
            return int(diff.days / 365)
        cu['age'] = cu.apply(age_apply, axis=1)
        return cu.drop(['DOB'], axis=1)

    def set_num_streams(self, cu, st, print_statement= False):
        if print_statement:
            print('set num streams, starting...')

        def num_streams_apply(row):
            stream_count = st.loc[st['customer_id'] == row['customer_id']]
            return len(stream_count)
        cu['num_streams'] = cu.apply(num_streams_apply, axis=1)
        return cu

    def set_sub_type(self, cu, st, print_statement= False):
        if print_statement:
            print('set sub type, starting...')

        def sub_type_apply(row):
            sub_type = st.loc[st['customer_id'] == row['customer_id']]
            return int(sub_type['invoice_item_id'].iloc[-1])
        cu['sub_type'] = cu.apply(sub_type_apply, axis=1)
        return cu
    
    def set_length_of_activity(self, cu, print_statement=False):
        if print_statement:
            print('set length of activity, starting...')

        def length_of_activity_apply(row):
            signup_date = datetime.datetime.strptime(row['signup_date'], '%Y-%m-%d').date()
            last_stream = datetime.datetime.strptime(row['last_stream'], '%Y-%m-%d').date()
            diff = last_stream - signup_date
            return diff.days
        cu['length_of_activity'] = cu.apply(length_of_activity_apply, axis=1)
        return cu

    def impute_state(self, cu, print_statement= False):
        if print_statement:
            print('impute state, starting...')
        cu['state'].fillna("singapore", inplace=True)
        return cu

    # End the pipeline - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# if __name__ == '__main__':
#     s, c = Aavail_Data_Ingestor(args.cpath, args.dpath, args.query, args.auto)
