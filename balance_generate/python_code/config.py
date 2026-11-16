import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from labbank_common.bank_config import get_bank

bank = get_bank()

report_date = None
balance_start_date = None
bs_side_map = {"E": "0","A": "1", "L": "2", "O": "3"}
rate_type_map = {"":"0", "F": "1", "V": "2", "A": "3"}

# Bank-specific generation parameters live in labbank_common/bank_config.py
avg_rate_map = bank.generation.avg_rate_map
stats_dict = bank.generation.stats_dict
margin_map = bank.generation.margin_map
