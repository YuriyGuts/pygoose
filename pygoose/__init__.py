"""
The main purpose of this package is shorthand access to frequently used experimentation functions.
Therefore, while things like global imports and hard-coded option values may look evil,
they are intentional in order to provide this particular simplified interface.
"""

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm as progressbar
from pkg_resources import get_distribution

from .kg import *


def _set_numpy_defaults():
    # Suppress scientific notation in NumPy float output.
    np.set_printoptions(suppress=True)


def _set_pandas_defaults():
    # Increase Pandas output limits.
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_colwidth', 2000)

    # Suppress scientific notation in Pandas float output.
    pd.set_option('display.float_format', lambda x: '%.6f' % x)


def _set_plotting_defaults():
    # Use higher resolution and larger fonts in plots.
    sns.set_context('talk')


def _console_main():
    # No need for argparse until we have more complex arguments.
    if len(sys.argv) > 1 and sys.argv[1] == 'init':
        kg.Project.init()


_set_numpy_defaults()
_set_pandas_defaults()
_set_plotting_defaults()


__all__ = [
    # Frequently used 3rd-party modules.
    'np',
    'pd',
    'plt',
    'sns',

    # Convenience modules.
    'progressbar',

    # Internal modules.
    'kg',
]

__version__ = get_distribution('pygoose').version
