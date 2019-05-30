import time
from functools import wraps

DELAY = 1
TRIES = 1000
BACKOFF = 2


def retry(f, *args, **kwargs):

    def retry_f(*args, **kwargs):
        delay_, tries_, backoff_ = DELAY, TRIES, BACKOFF
        while tries_ > 1:
            try:
                return f(*args, **kwargs)
            except Exception as e:
                print('Caught exception %s, Retrying, %s times left' % (e, tries_))
                # time.sleep(delay_)
                tries_ -= 1
                delay_ *= backoff_

        return f(*args, **kwargs)

    return retry_f