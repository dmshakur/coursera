"""
Using model.py and ./unittests/ModelTests.py as an example create logger.py and ./unittests/LoggerTests.py.

Modify the files so that there are at a minimum the following tests:

ensure predict log is automatically created
ensure train log is automatically created
ensure that train log archives last used training data
ensure that 'n' predictions result in 'n' log entries
ensure that predict gracefully handles NaNs
IMPORTANT: when writing to a log file from a unit
    test you will want to ensure that you do not
    modify or delete existing 'production' logs.
    You can test your function with the following
    code (although it is likely easier to work
    directly in a terminal).
"""

import functools
import os
import time
import numpy as np
# from model import MODEL_VERSION
MODEL_VERSION = 0.1


def set_log_path(production=False):
    LOG_PATH = 'logs/production' if production else 'logs/test'


CWD, LOG_PATH, LOG_FILE = '.', '', '{}.log'
LOG_TYPES = {'T': 'train', 'P': 'predict', 'D': 'train_data'}
TRAIN_HEADER = ['time_stamp', 'model_version', 'run_time']
PREDICT_HEADER = ['time_stamp', 'model_version', 'run_time', 'prediction_prob']
RUN_STATUS = {'invalid_input': 'input was invalid: {}',
              'valid_input': 'valid_input'}

set_log_path()


class BadArgumentError(Exception):
    """
    An exception that's sole purpose is to notify the user
    that an argument was of incorrect format.
    """

    def __init__(self, arg='', entry='', message='{} of type {} is not a valid argument. {}'):
        self.arg = arg
        self.entry = entry
        self.message = message
        super().__init__(self.message.format(self.arg, type(self.arg), self.entry))


def create_log_entry(data):
    """
    This function is made to take as arguments a
    list of items that will make up a log entry or
    header for the logs.
    """

    if '__iter__' not in data.__dir__():
        return BadArgumentError(type(data))

    log_entry = str()

    for point in data:

        if '__iter__' in point.__dict__():
            joined_point = ','.join(point)
            log_entry += str(joined_point)
        else:
            log_entry += str(point) + ','

    return log_entry[:-1]


def fetch_data_logger(fn):
    """
    A decorator function made for creating a log for the training data.
    It does not create a new log entry, it only overwrites so that there is only
    the most recently used training data as log to prevent overuse of resources.
    """
    @functools.wraps(fn)
    def wrapper():

        data = fn()

        create_or_update_log(data, LOG_TYPES['D'])

        return data
    return wrapper


def create_or_update_log(log_type, log_entry, header=False):
    """
    Creates or updates the log.
    `log_type` specifies if it is a train, data, or predict log.
    `log_entry` is an iterable with the data that will be logged.
    `header` is the header for the log file if necessary.
    """
    if log_type not in list(LOG_TYPES.values()):
        raise BadArgumentError(log_type, log_entry)

    if '__iter__' not in log_entry.__dir__():
        raise TypeError(
            f'Data needs to be of iterable type, type {type(log_entry)} received')

    log_file_path = os.path.join(CWD, LOG_PATH, LOG_FILE.format(log_type))

    if log_type == 'train_data':
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
        log_file = open(log_file_path, 'w')
        log_file.write(log_entry)
        log_file.close()
        return None

    file_mode = 'a' if header else 'w+'

    for i in range(len(log_entry)):
        log_entry[i] = str(log_entry[i])

    log_file = open(log_file_path, file_mode)

    if header:
        log_file.write(header)

    for entry in log_entry:
        log_file.write(entry)

    log_file.close()


def predict_logger(model_fn):
    """
    A decorator function for a trained model.
    It's purpose is to take data from the model and log it.
    It logs the following: time stamp, model version, run time,
    prediction and prediction probability
    """
    @functools.wraps(model_fn)
    def wrapper(*args, **kwargs):

        timer_start = time.perf_counter()
        try:
            predictions = model_fn(*args, **kwargs)
        except:
            timer_end = time.perf_counter()
            run_time = timer_end - timer_start
            time_stamp = time.localtime()
            prediction_prob = np.nan
            run_status = RUN_STATUS['invalid_input']

            log_entry = [time_stamp, MODEL_VERSION,
                         run_time, prediction_prob, run_status]
            create_or_update_log(LOG_TYPES['P'], log_entry)
            print('Logging: predict logger')

            return None

        timer_end = time.perf_counter()

        run_time = timer_end - timer_start
        model_version = MODEL_VERSION
        prediction_prob = sum(predictions.y_proba) / len(predictions.y_proba)
        time_stamp = time.localtime()

        log_entry = [time_stamp, MODEL_VERSION, run_time, prediction_prob]
        create_or_update_log(LOG_TYPES['P'], log_entry)

        print('Logging: predict logger')

        return predictions
    return wrapper


def train_logger(model_fn):
    """
    A decorator function for training models.
    It's purpose is to take data from the model and log it.
    It logs the following: time stamp, model version and run time.
    """

    @functools.wraps(model_fn)
    def wrapper(*args, **kwargs):
        timer_start = time.perf_counter()
        model = model_fn(*args, **kwargs)
        timer_end = time.perf_counter()

        time_stamp = time.localtime()
        model_version = MODEL_VERSION
        run_time = timer_end - timer_start

        log_entry = [time_stamp, MODEL_VERSION, run_time]

        header = \
            ','.join(TRAIN_HEADER) \
            if os.path.exists(CWD, LOG_PATH, LOG_FILE.format(LOG_TYPES['T'])) \
            else False

        create_or_update_log(LOG_TYPES['T'], log_entry, header)

        print('Logging: train logger')

        return model
    return wrapper
