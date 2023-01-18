import pyspark.sql.functions as F

from pyspark.sql import DataFrame, WindowSpec
from pyspark.sql import Window
from typing import Callable


def create_window_frame_ever(partition_key: str, order_key: str):
    window = (
        Window()
        .partitionBy(partition_key)
        .orderBy(order_key)
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    return window, f"{partition_key}_ever"


def create_window_frame_hour_interval(partition_key: str, order_key: str, interval: int):
    def to_hours(h): return h * 3600
    window = (
        Window()
        .partitionBy(partition_key)
        .orderBy(F.col(order_key).cast("timestamp").cast("long"))
        .rangeBetween(-to_hours(interval), Window.currentRow)
    )
    return window, f"{partition_key}_last_{interval}h"


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
