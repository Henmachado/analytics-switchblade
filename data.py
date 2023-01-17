from features import create_sum_features_type_ever

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

spark = SparkSession.builder.getOrCreate()

schema = StructType([
    StructField("player_name", StringType()),
    StructField("team_name", StringType()),
    StructField("player_points", IntegerType()),
    StructField("player_assists", IntegerType()),
])
data = [
    ('James', "Lakers", 28, 14),
    ('Irving', "Nets", 23, 13),
    ('Durant', "Nets", 35, 8),
    ('Curry', "Warriors", 38, 9),
    ('Harden', "Nets", 19, 12)
    ]

spark_df = spark.createDataFrame(data=data, schema=schema)

dff = create_sum_features_type_ever(spark_df, ["team_name"], ["player_points", "player_assists"])

dff.printSchema()
dff.show(truncate=False)
