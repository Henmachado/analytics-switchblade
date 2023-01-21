from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window


def generate_window_frame_ever(partition_key: str, order_key: str) -> list:
    return [(
        Window()
        .partitionBy(partition_key)
        .orderBy(order_key)
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )]


def generate_window_frame_interval(partition_key: str, order_key: str, interval_list: list) -> list:
    def to_hours(h): return h * 3600
    _interval_list = []
    for interval in interval_list:
        w = (
            Window()
            .partitionBy(partition_key)
            .orderBy(F.col(order_key).cast("timestamp").cast("long"))
            .rangeBetween(-to_hours(interval), Window.currentRow)
        )
        _interval_list.append(w)
    return _interval_list


def generate_count_feature(df: DataFrame, target_colum: str, wf_intervals: list) -> DataFrame:
    _df = df
    for interval in wf_intervals:
        _df = _df.withColumn(f"{interval}", F.count(target_colum).over(interval))
    return _df


def generate_sum_feature(df: DataFrame, target_colum: str, wf_intervals: list) -> DataFrame:
    _df = df
    for interval in wf_intervals:
        _df = _df.withColumn(f"{interval}", F.sum(target_colum).over(interval))
    return _df
