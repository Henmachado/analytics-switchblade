import datetime

from itertools import product

from features import *

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType


def main() -> None:
    spark = SparkSession.builder.getOrCreate()

    schema = StructType([
        StructField("player_name", StringType()),
        StructField("team_name", StringType()),
        StructField("player_points", IntegerType()),
        StructField("player_assists", IntegerType()),
        StructField("date", DateType()),
    ])

    data = [
        ('James', "Lakers", 28, 14, datetime.datetime.strptime('2021-01-01', "%Y-%m-%d").date()),
        ('Irving', "Nets", 23, 13, datetime.datetime.strptime('2021-01-01', "%Y-%m-%d").date()),
        ('Durant', "Nets", 35, 8, datetime.datetime.strptime('2021-01-02', "%Y-%m-%d").date()),
        ('Curry', "Warriors", 38, 9, datetime.datetime.strptime('2021-01-01', "%Y-%m-%d").date()),
        ('Harden', "Nets", 19, 12, datetime.datetime.strptime('2021-01-03', "%Y-%m-%d").date())
    ]

    spark_df = spark.createDataFrame(data=data, schema=schema)

    targets = {
        "player_points": (
            ["team_name", "player_name"],
            [24, 48, 72],
            ["date"]
        ),
        "player_assists": (
            ["team_name", "player_name"],
            [24],
            ["date"]
        )
    }

    features = list(targets.keys())
    for feature in features:
        x, y, z = targets.get(feature)
        specs = list(product(x, y, z))
        spark_df = generate_sum_feature(spark_df, feature, specs)
        spark_df = generate_count_feature(spark_df, feature, specs)

    return spark_df.show()


if __name__ == "__main__":
    main()

