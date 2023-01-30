from functools import reduce
from itertools import product
from typing import Callable

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window


def generate_hour_window_spec(specs: tuple) -> tuple:
    window_spec = (
        Window()
        .partitionBy(specs[0])
        .orderBy(F.col(specs[2]).cast("timestamp").cast("long"))
        .rangeBetween(-specs[1] * 3600, Window.currentRow)
    )
    feature_name = f"per_{specs[0]}_{specs[1]}h_orderby_{specs[2]}"
    return window_spec, feature_name


def generate_ever_window_spec(specs: tuple) -> tuple:
    window_spec = (
        Window()
        .partitionBy(specs[0])
        .orderBy(F.col(specs[2]).cast("timestamp").cast("long"))
        .rangeBetween(Window.unboundedPreceding, Window.currentRow)
    )
    feature_name = f"per_{specs[0]}_{specs[1]}h_orderby_{specs[2]}"
    return window_spec, feature_name


def generate_window_spec_list(specs: list[tuple], window_type: Callable[[tuple], tuple]) -> list:
    window_spec_list = []
    for spec in specs:
        w = window_type(spec)
        window_spec_list.append(w)
    return window_spec_list


def generate_count_feature(df: DataFrame, target_colum: str, specs: list[tuple]) -> DataFrame:
    return reduce(
        lambda _df, spec: _df.withColumn(
            f"count_{target_colum}_{spec[1]}", F.count(target_colum).over(spec[0])
        ),
        specs,  # iterable
        df  # initial value of the iteration
    )


def generate_count_distinct_feature(df: DataFrame, target_colum: str, specs: list[tuple]) -> DataFrame:
    return reduce(
        lambda _df, spec: _df.withColumn(
            f"count_distinct_{target_colum}_{spec[1]}", F.approx_count_distinct(target_colum).over(spec[0])
        ),
        specs,  # iterable
        df  # initial value of the iteration
    )


def generate_sum_feature(df: DataFrame, target_colum: str, specs: list[tuple]) -> DataFrame:
    return reduce(
        lambda _df, spec: _df.withColumn(
            f"sum_{target_colum}_{spec[1]}", F.sum(target_colum).over(spec[0])
        ),
        specs,  # iterable
        df  # initial value of the iteration
    )


def generate_features(spark_df: DataFrame, features: dict) -> DataFrame:
    feature_names = list(features.keys())
    for feature in feature_names:
        x, y, z = features.get(feature)
        specs = list(product(x, y, z))
        window_specs_hour = generate_window_spec_list(specs, generate_hour_window_spec)
        window_specs_ever = generate_window_spec_list(specs, generate_ever_window_spec)
        spark_df = generate_sum_feature(spark_df, feature, window_specs_hour)
        spark_df = generate_sum_feature(spark_df, feature, window_specs_ever)
        spark_df = generate_count_feature(spark_df, feature, window_specs_hour)
        spark_df = generate_count_feature(spark_df, feature, window_specs_ever)
        spark_df = generate_count_distinct_feature(spark_df, feature, window_specs_ever)
    return spark_df
