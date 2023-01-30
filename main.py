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

    df = spark.createDataFrame(data=data, schema=schema)

    features = {
        "player_points": (
            ["team_name", "player_name"],
            [24, 48],
            ["date"]
        ),
        "player_assists": (
            ["team_name"],
            [24],
            ["date"]
        )
    }

    spark_df = generate_features(spark_df=df, features=features)

    return spark_df.show()


if __name__ == "__main__":
    main()

