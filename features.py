from pyspark.sql import DataFrame


def create_sum_features_type_ever(df: DataFrame, feature_list: list, sum_column: str) -> DataFrame:
    for feature in feature_list:
        return df.selectExpr(
            "*",
            f"sum({sum_column}) over(partition by {feature}) as ft_sum_total_{feature}_ever"
        )
