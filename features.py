import pyspark.sql.functions as F

from pyspark.sql import DataFrame
from pyspark.sql import Window


def create_sum_features_type_ever(
        df: DataFrame, feature_list: list, sum_column: str
) -> DataFrame:
    _df = df
    for feature in feature_list:
        window = (
            Window()
            .partitionBy(feature)
            .rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )
        _df = df.withColumn(
            f"ft_sum_total_{feature}_ever",
            F.sum(F.col(sum_column)).over(window)
        )
    return _df