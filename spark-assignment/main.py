from pyspark.sql import SparkSession
import pyspark.sql.types as T
import pyspark.sql.functions as func
from pyspark.sql import Row
from pyspark.sql.functions import last,col
from pyspark.sql.window import Window



def load_trades(spark):
    data = [
        (10, 1546300800000, 37.50, 100.000),
        (10, 1546300801000, 37.51, 100.000),
        (20, 1546300804000, 12.67, 300.000),
        (10, 1546300807000, 37.50, 200.000),
    ]
    schema = T.StructType(
        [
            T.StructField("id", T.LongType()),
            T.StructField("timestamp", T.LongType()),
            T.StructField("price", T.DoubleType()),
            T.StructField("quantity", T.DoubleType()),
        ]
    )

    return spark.createDataFrame(data, schema)


def load_prices(spark):
    data = [
        (10, 1546300799000, 37.50, 37.51),
        (10, 1546300802000, 37.51, 37.52),
        (10, 1546300806000, 37.50, 37.51),
    ]
    schema = T.StructType(
        [
            T.StructField("id", T.LongType()),
            T.StructField("timestamp", T.LongType()),
            T.StructField("bid", T.DoubleType()),
            T.StructField("ask", T.DoubleType()),
        ]
    )

    return spark.createDataFrame(data, schema)


def fill(trades, prices):
    """
    Combine the sets of events and fill forward the value columns so that each
    row has the most recent non-null value for the corresponding id. For
    example, given the above input tables the expected output is:

    +---+-------------+-----+-----+-----+--------+
    | id|    timestamp|  bid|  ask|price|quantity|
    +---+-------------+-----+-----+-----+--------+
    | 10|1546300799000| 37.5|37.51| null|    null|
    | 10|1546300800000| 37.5|37.51| 37.5|   100.0|
    | 10|1546300801000| 37.5|37.51|37.51|   100.0|
    | 10|1546300802000|37.51|37.52|37.51|   100.0|
    | 20|1546300804000| null| null|12.67|   300.0|
    | 10|1546300806000| 37.5|37.51|37.51|   100.0|
    | 10|1546300807000| 37.5|37.51| 37.5|   200.0|
    +---+-------------+-----+-----+-----+--------+

    :param trades: DataFrame of trade events
    :param prices: DataFrame of price events
    :return: A DataFrame of the combined events and filled.
    """
    id_window = Window.partitionBy('id').orderBy('timestamp').rowsBetween(Window.unboundedPreceding, 0)


    df = trades \
        .withColumn("bid", func.lit(None)) \
        .withColumn("ask", func.lit(None)) \
        .unionByName(
        prices
            .withColumn("price", func.lit(None))
            .withColumn("quantity", func.lit(None))
    ) \
        .withColumn("bid", func.last("bid", True).over(id_window)) \
        .withColumn("ask", func.last("ask", True).over(id_window)) \
        .withColumn("price", func.last("price", True).over(id_window)) \
        .withColumn("quantity", func.last("quantity", True).over(id_window)) \
        .orderBy("timestamp") \
        .select("id", "timestamp", "bid", "ask", "price", "quantity")

    return df



def pivot(trades, prices,spark):
    """
    Pivot and fill the columns on the event id so that each row contains a
    column for each id + column combination where the value is the most recent
    non-null value for that id. For example, given the above input tables the
    expected output is:

    +---+-------------+-----+-----+-----+--------+------+------+--------+-----------+------+------+--------+-----------+
    | id|    timestamp|  bid|  ask|price|quantity|10_bid|10_ask|10_price|10_quantity|20_bid|20_ask|20_price|20_quantity|
    +---+-------------+-----+-----+-----+--------+------+------+--------+-----------+------+------+--------+-----------+
    | 10|1546300799000| 37.5|37.51| null|    null|  37.5| 37.51|    null|       null|  null|  null|    null|       null|
    | 10|1546300800000| null| null| 37.5|   100.0|  37.5| 37.51|    37.5|      100.0|  null|  null|    null|       null|
    | 10|1546300801000| null| null|37.51|   100.0|  37.5| 37.51|   37.51|      100.0|  null|  null|    null|       null|
    | 10|1546300802000|37.51|37.52| null|    null| 37.51| 37.52|   37.51|      100.0|  null|  null|    null|       null|
    | 20|1546300804000| null| null|12.67|   300.0| 37.51| 37.52|   37.51|      100.0|  null|  null|   12.67|      300.0|
    | 10|1546300806000| 37.5|37.51| null|    null|  37.5| 37.51|   37.51|      100.0|  null|  null|   12.67|      300.0|
    | 10|1546300807000| null| null| 37.5|   200.0|  37.5| 37.51|    37.5|      200.0|  null|  null|   12.67|      300.0|
    +---+-------------+-----+-----+-----+--------+------+------+--------+-----------+------+------+--------+-----------+

    :param trades: DataFrame of trade events
    :param prices: DataFrame of price events
    :return: A DataFrame of the combined events and pivoted columns.
    """

    data = [
        (10, 1546300800000,None, None, 37.5, 100.000),
        (10, 1546300801000,None, None, 37.51, 100.000),
        (20, 1546300804000,None, None, 12.67, 300.000),
        (10, 1546300807000,None, None, 37.5, 200.000),
        (10, 1546300799000, 37.5, 37.51,None, None),
        (10, 1546300802000, 37.51, 37.52,None, None),
        (10, 1546300806000, 37.5, 37.51,None, None),
    ]
    schema = T.StructType(
        [
            T.StructField("id", T.LongType()),
            T.StructField("timestamp", T.LongType()),
            T.StructField("bid", T.DoubleType()),
            T.StructField("ask", T.DoubleType()),
            T.StructField("price", T.DoubleType()),
            T.StructField("quantity", T.DoubleType()),
        ]
    )
    new_df = spark.createDataFrame(data, schema)

    new_df1 = new_df.groupBy('id','timestamp', 'bid', 'ask', 'price', 'quantity').pivot('id').agg(
            func.last("bid").alias('bid'),
            func.last("ask").alias('ask'),
            func.last('price').alias('price'),
            func.last('quantity').alias('quantity')) \
        .orderBy("timestamp")
    cols = new_df1.columns[6:]

    ffill = Window.orderBy('timestamp').rowsBetween(Window.unboundedPreceding, Window.currentRow)
    sdf1 = new_df1.select('id', 'timestamp', 'bid', 'ask', 'price', 'quantity',
                          *[last(col(c), ignorenulls=True).over(ffill).alias(c) for c in cols])

    print("below is second approach")
    id_window = Window.partitionBy('timestamp').orderBy('timestamp').rowsBetween(Window.unboundedPreceding, 0)

    df1 = trades \
        .withColumn("bid", func.lit(None)) \
        .withColumn("ask", func.lit(None)) \
        .unionByName(
        prices
            .withColumn("price", func.lit(None))
            .withColumn("quantity", func.lit(None))
    ) \
        .withColumn("bid", func.last("bid", True).over(id_window)) \
        .withColumn("ask", func.last("ask", True).over(id_window)) \
        .withColumn("price", func.last("price", True).over(id_window)) \
        .withColumn("quantity", func.last("quantity", True).over(id_window)) \
        .orderBy("timestamp") \
        .select("id", "timestamp", "bid", "ask", "price", "quantity")

    df = df1.groupBy('id','timestamp', 'bid', 'ask', 'price', 'quantity').pivot('id').agg(
            func.last("bid",ignorenulls=True).alias('bid'),
            func.last("ask",ignorenulls=True).alias('ask'),
            func.last('price',ignorenulls=True).alias('price'),
            func.last('quantity',ignorenulls=True).alias('quantity')) \
        .orderBy("timestamp")

    cols = df.columns[6:]

    ffill1 = Window.orderBy('timestamp').rowsBetween(Window.unboundedPreceding, Window.currentRow)
    df = df.select('id', 'timestamp', 'bid', 'ask', 'price', 'quantity',
                          *[last(col(c), ignorenulls=True).over(ffill1).alias(c) for c in cols])


    return df


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    trades = load_trades(spark)
    trades.show()

    prices = load_prices(spark)
    prices.show()

    fill(trades, prices).show()

    pivot(trades, prices,spark).show()
