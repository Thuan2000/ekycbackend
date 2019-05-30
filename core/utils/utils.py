from datetime import datetime
import functools


def timestamp_to_datetime(timestamp, time_zone=7):
    return datetime.utcfromtimestamp(timestamp+time_zone*3600).strftime('%Y-%m-%d %H:%M:%S')


def recursive_retry(function):
    @functools.wraps(function)
    def retry(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except:
            print('retry', function)
    return retry


def get_class_attributes(class_, exclude=[]):
    accepted_attributes = []
    exclude.append('__')  # by default we do not get python navite attr
    attributes = dir(class_)
    for attr in attributes:
        if not is_contain_substrings(attr, exclude):
            accepted_attributes.append(attr)
    values = [getattr(class_, attr) for attr in accepted_attributes]
    return values


def is_contain_substrings(str_, substrs):
    for substr in substrs:
        if substr in str_:
            return True
    return False
