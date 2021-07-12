import logging
import time
from functools import update_wrapper


class my_logger(object):
    """
    Decorator that logs the function input arguments into a log file 
    named with the function name.
    """

    def __init__(self, orig_func):
        self.orig_func = orig_func

        logging.basicConfig(
            filename="{}.log".format(orig_func.__name__), level=logging.INFO
        )
        update_wrapper(
            self, orig_func
        )  # Update a wrapper function to look like the wrapped function

    def __call__(self, *args, **kwargs):
        logging.info("Ran with args: {}, and kwargs: {}".format(args, kwargs))
        return self.orig_func(*args, **kwargs)


class my_timer(object):
    """
    Decorator that displays the function's running time.
    """

    def __init__(self, orig_func):
        self.orig_func = orig_func
        update_wrapper(
            self, orig_func
        )  # Update a wrapper function to look like the wrapped function

    def __call__(self, *args, **kwargs):
        t1 = time.time()
        result = self.orig_func(*args, **kwargs)
        t2 = time.time() - t1
        print("{} ran in: {} sec".format(self.orig_func.__name__, t2))
        return result


class my_cache(object):
    """
    Decorator that caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned, and
    not re-evaluated.
    """

    def __init__(self, orig_func):
        self.orig_func = orig_func
        self.cache = {}
        update_wrapper(
            self, orig_func
        )  # Update a wrapper function to look like the wrapped function

    def __call__(self, *args, **kwargs):
        try:
            return self.cache[args, kwargs]
        except KeyError:
            self.cache[args, kwargs] = self.orig_func(*args, **kwargs)
        except TypeError:
            # uncacheable -- for instance, passing a list as an argument.
            # Better to not cache than to blow up entirely.
            return self.orig_func(*args, **kwargs)

    def __repr__(self):
        return self.orig_func.__repr__()
