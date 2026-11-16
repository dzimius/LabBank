import os
import sys

import pandas as pd

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from labbank_common.bank_config import get_bank

report_date = pd.Timestamp(get_bank().report_date)
