from pyspark.sql import DataFrame, WindowSpec
from pyspark.sql import functions as F
from pyspark.sql import Window

from dataclasses import dataclass
from typing import Callable


@dataclass
class WindFuncFeatureGenerator:
    df: DataFrame
    column_ref: str
    partition_key: str
    order_key: str
    interval: int


def create_window_frame_ever(params: WindFuncFeatureGenerator):
    window = (
        Window()
        .partitionBy(params.partition_key)
        .orderBy(params.order_key)
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    return window, f"{params.partition_key}_ever"


def create_window_frame_hour_interval(params: WindFuncFeatureGenerator):
    def to_hours(h): return h * 3600
    window = (
        Window()
        .partitionBy(params.partition_key)
        .orderBy(F.col(params.order_key).cast("timestamp").cast("long"))
        .rangeBetween(-to_hours(params.interval), Window.currentRow)
    )
    return window, f"{params.partition_key}_last_{params.interval}h"


def create_count_feature(
        df: DataFrame, window: Callable[[str], tuple[WindowSpec, str]], count_column: str) -> DataFrame:
    _df = df
    return _df.withColumn(
        f"ft_total_count_{count_column}_per_{window[1]}", F.sum(F.lit(1)).over(window[0])
    )


def create_sum_feature(
        df: DataFrame, window: Callable[[str], tuple[WindowSpec, str]], sum_column: str) -> DataFrame:
    _df = df
    return _df.withColumn(
        f"ft_total_sum_{sum_column}_per_{window[1]}", F.sum(F.col(sum_column)).over(window[0])
    )
