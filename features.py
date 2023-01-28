from functools import reduce
from itertools import product

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window


def generate_window_spec(specs: tuple) -> tuple:
    window_spec = (
        Window()
        .partitionBy(specs[0])
        .orderBy(F.col(specs[2]).cast("timestamp").cast("long"))
        .rangeBetween(-specs[1] * 3600, Window.currentRow)
    )
    feature_name = f"per_{specs[0]}_{specs[1]}h_orderby_{specs[2]}"
    return window_spec, feature_name


def generate_window_spec_list(specs: list[tuple]) -> list:
    window_spec_list = []
    for spec in specs:
        w = generate_window_spec(spec)
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


def generate_sum_feature(df: DataFrame, target_colum: str, specs: list[tuple]) -> DataFrame:
    return reduce(
        lambda _df, spec: _df.withColumn(
            f"sum_{target_colum}_{spec[1]}", F.sum(target_colum).over(spec[0])
        ),
        specs,  # iterable
        df  # initial value of the iteration
    )


def generate_features(spark_df: DataFrame, targets: dict) -> DataFrame:
    features = list(targets.keys())
    for feature in features:
        x, y, z = targets.get(feature)
        specs = list(product(x, y, z))
        window_specs = generate_window_spec_list(specs)
        spark_df = generate_sum_feature(spark_df, feature, window_specs)
        spark_df = generate_count_feature(spark_df, feature, window_specs)
    return spark_df
