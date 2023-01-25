from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window


def to_hours(h):
    return h * 3600


def generate_window_object(specs: tuple) -> tuple:
    w = (
        Window()
        .partitionBy(specs[0])
        .orderBy(F.col(specs[2]).cast("timestamp").cast("long"))
        .rangeBetween(-to_hours(specs[1]), Window.currentRow)
    )
    ft_name = f"per_{specs[0]}_{specs[1]}h_orderby_{specs[2]}"
    return w, ft_name


def generate_window_objects_list(specs: list[tuple]) -> list:
    _wf = []
    for spec in specs:
        w = generate_window_object(spec)
        _wf.append(w)
    return _wf


def generate_count_feature(df: DataFrame, target_colum: str, specs: list[tuple]) -> DataFrame:
    _df = df
    for spec in generate_window_objects_list(specs):
        _df = _df.withColumn(
            f"count_{target_colum}_{spec[1]}", F.count(target_colum).over(spec[0])
        )
    return _df


def generate_sum_feature(df: DataFrame, target_colum: str, specs: list[tuple]) -> DataFrame:
    _df = df
    for spec in generate_window_objects_list(specs):
        _df = _df.withColumn(
            f"sum_{target_colum}_{spec[1]}", F.sum(target_colum).over(spec[0])
        )
    return _df
