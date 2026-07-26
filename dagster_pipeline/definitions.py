from dagster import Definitions

from dagster_pipeline.assets import all_assets
from dagster_pipeline.jobs import all_jobs

defs = Definitions(
    assets=all_assets,
    jobs=all_jobs,
)
