from features import create_sum_features_type_ever

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

spark = SparkSession.builder.getOrCreate()

schema = StructType([
    StructField("player_name", StringType()),
    StructField("team_name", StringType()),
    StructField("player_points", IntegerType()),
])
data = [
    ('James', "Lakers", 28),
    ('Irving', "Nets", 23),
    ('Durant', "Nets", 35),
    ('Curry', "Warriors", 38),
    ('Harden', "Nets", 19)
    ]

spark_df = spark.createDataFrame(data=data, schema=schema)

dff = create_sum_features_type_ever(spark_df, ["team_name"], "player_points")

dff.printSchema()
dff.show(truncate=False)
