"""One static partition per bank (labbank_common/bank_config.BANKS).

Every asset is partitioned by bank; the runner passes the partition key to each
workflow subprocess as LABBANK_BANK, which selects the bank's SQL database and its
input/output folders. Materialize a job for "pln", "eur", or both.
"""
from dagster import StaticPartitionsDefinition

from labbank_common.bank_config import BANKS

BANK_PARTITIONS = StaticPartitionsDefinition(sorted(BANKS))
