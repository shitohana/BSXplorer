import os

import polars
import multiprocessing


def set_polars_threads(threads: int = 1):
    """
    Set number of threads for polars tabular operations

    Parameters
    ----------
    threads
        number of threads

    See Also
    --------
    `polars.thread_pool_size <https://docs.pola.rs/api/python/stable/reference/api/polars.thread_pool_size.html#polars.thread_pool_size>`_
    """
    if threads <= multiprocessing.cpu_count():
        os.environ["POLARS_MAX_THREADS"] = str(threads)
    else:
        raise ValueError(
            f"Selected more threads ({threads}) than actual cores ({multiprocessing.cpu_count()}) on the machine!")


def _enable_string_cache():
    polars.enable_string_cache()
