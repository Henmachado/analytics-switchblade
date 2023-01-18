import datetime

from features import *

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType

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

dff = create_sum_feature(
    spark_df,
    create_window_frame_hour_interval(partition_key="team_name", order_key="date", interval=24),
    sum_column="player_assists"
)

dff.printSchema()
dff.show(truncate=False)
