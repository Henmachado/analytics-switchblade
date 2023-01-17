import itertools
import pyspark.sql.functions as F

from pyspark.sql import DataFrame
from pyspark.sql import Window


def create_sum_features_type_ever(df: DataFrame, feature_list: list, sum_columns: list) -> DataFrame:
    _df = df
    for feature, sum_column in itertools.product(feature_list, sum_columns):
        window = (
            Window().partitionBy(feature).rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )
        _df = _df.withColumn(
            f"ft_sum_total_{sum_column}_per_{feature}_ever", F.sum(F.col(sum_column)).over(window)
        )
    return _df
