#!/usr/bin/env python
'''
logger tests
'''

import unittest
import os
import numpy as np
import pandas as pd
from model import *

CWD, LOG_PATH, LOG_FILE = '.', 'logs/test', '{}.log'
LOG_TYPES = {'T': 'train', 'P': 'predict', 'D': 'train_data'}

class LoggerTest(unittest.TestCase):

    def test_01_predict_log_creation(self):
        '''
        test for nan values
        '''
        predict_log_path = os.path.join(CWD, LOG_PATH, LOG_FILE.format(LOG_TYPES['P']))
        query = [
            np.array([[6.1, 2.8]]), 
            np.array([[7.7, 2.5]]),
            np.array([[5.8, 3.8]])
        ]

        predictions = model_predict(query)

        self.assertTrue(
            os.path.exists(predict_log_path),
            msg = 'Asserting: prediction log has been created'
        )


    def test_02_train_log_creation(self):
        '''
        check if train log was automatically created
        '''
        train_log_path = os.path.join(CWD, LOG_PATH, LOG_FILE.format(LOG_TYPES['T']))

        model_train()

        self.assertTrue(
            os.path.exists(train_log_path),
            msg= 'Asserting: train log has been created'
        )


    def test_03_train_logs_archived(self):
        '''
        Check that training logs are archived
        '''
        train_log_path = os.path.join(CWD, LOG_PATH, LOG_FILE.format(LOG_TYPES['D']))

        model_train()
        
        self.assertTrue(
            os.path.exists(train_log_path), 
            msg='Asserting: most recent training data is logged'
        )


    def test_04_pred_and_log_entry_count(self):
        '''
        check that the 'n' predictions and 'n'
        log entries are equal in count
        '''
        predict_log_path = os.path.join(CWD, LOG_PATH, LOG_PATH.format(LOG_TYPES['P']))
        query = [
            np.array([[6.1, 2.8]]), 
            np.array([[7.7, 2.5]]), 
            np.array([[5.8, 3.8]])
        ]
        predictions = model_predict(query)

        # Check that n probability predictions are equal to n log entries
            
        log = pd.read_csv(predict_log_path, header = True)

        self.assertEqual(
            proba_len,
            len(log['prediction_proba']), 
            msg='Asserting: log entries == probability predictions'
        )


    def test_05_check_nan(self):
        '''
        Check if there are any NaN values
        '''
        train_df = pd.read_csv(os.path.join(CWD, LOG_PATH, LOG_FILE.format(LOG_TYPES['T'])))
        predict_df = pd.read_csv(os.path.join(CWD, LOG_PATH, LOG_FILE.format(LOG_TYPES['P'])))

        self.assertFalse(
            train_df.isnull().values.any(),
            msg = 'Asserting: training logs don\'t have any  NaN values'
        )
        self.assertFalse(
            predict_df.isnull().values.any(),
            msg = 'Asserting: prediction logs don\'t have any  NaN values'
        )


if __name__ == '__main__':
    unittest.main()
