#!/usr/bin/env python
'''
logger tests
'''

import unittest
from logger import *
from model import *


set_log_path(production=False)


def get_model():
    try:
        model = model_load()
    except:
        model = model_train()


class LoggerTest(unittest.TestCase):

    def test_01_check_pred_log(self):
        '''
        test for nan values
        '''
        model = get_model()

        query = np.array([np.nan, 10, 12, 0.5])

        model_predict(query)

        log_file_path = os.path.join(
            CWD, LOG_PATH, LOG_FILE.format(LOG_TYPES['P']))
        log_file = open(log_file_path, 'r')

        last_line = log_file.readlines()[-1]
        log_file.close()

        self.assertIn(RUN_STATUS['invalid_input'], last_line)

    def test_02_check_train_log(self):
        '''
        check if train log was automatically created
        '''
        model = model_train()

        log_file_path = os.path.join(
            CWD, LOG_PATH, LOG_FILE.format(LOG_TYPES['T']))

        self.assertTrue(os.path.exists(log_file_path))

    def test_03_train_logs_archived(self):
        '''
        Check that training logs are archived
        '''
        model = model_train()

        log_file_path = os.path.join(
            CWD, LOG_PATH, LOG_FILE.format(LOG_TYPES['D']))

        self.assertTrue(os.path.exists(log_file_path))

    def test_04_pred_and_log_entry_count(self):
        '''
        check that the 'n' predictions and 'n'
        log entries are equal in count
        '''
        model = get_model()

        for i in range(5):
            query_set = np.random.rand(i, i)
            query_set_len = len(query_set)
            pred = model_predict(query_set)
            y_pred = len(pred['y_pred'])
            y_proba = len(pred['y_proba'])

            self.assertEqual(query_set_len, y_pred)
            self.assertEqual(query_set_len, y_proba)


if __name__ == '__main__':
    unittest.main()
