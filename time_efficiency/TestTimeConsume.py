import os
import sys
import numpy as np
import nltk
import time


# Set the path for spark installation
# this is the path where you have built spark using sbt/sbt assembly
os.environ['SPARK_HOME'] = "/Applications/spark-2.1.0"
# os.environ['SPARK_HOME'] = "/home/jie/d2/spark-0.9.1"
# Append to PYTHONPATH so that pyspark could be found
sys.path.append("/Applications/spark-2.1.0/python")

os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--packages org.mongodb.spark:mongo-spark-connector_2.10:2.0.0 --driver-memory 5g  --executor-memory 5g pyspark-shell")


# Now we are ready to import Spark Modules
try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.ml import Pipeline
    from pyspark.ml.linalg import Vectors, VectorUDT
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import *
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.sql import Row
    from pyspark.sql.functions import *
    from pyspark.sql.types import *
    from pyspark.sql import SparkSession
    from pyspark.mllib.linalg import SparseVector, DenseVector
    from pyspark.ml.feature import CountVectorizer
    from pyspark.ml.clustering import LDA
    from pyspark.ml.feature import Word2Vec
    from pyspark.mllib.linalg import Vectors
    from pyspark.ml.feature import Tokenizer, RegexTokenizer
    from pyspark.sql.functions import col, udf
    from pyspark.sql.types import IntegerType
    from pyspark.ml.feature import StopWordsRemover
    from pyspark.ml.feature import HashingTF, IDF, Tokenizer
    from pyspark.ml import Pipeline
    from pyspark.ml.regression import RandomForestRegressor
    from pyspark.ml.feature import VectorIndexer
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.sql.functions import udf
    from pyspark.sql.types import IntegerType, FloatType
    from pyspark.sql.window import Window
    from pyspark.sql.functions import rank, col

except ImportError as e:
    print("Error importing Spark Modules", e)
    sys.exit(1)

sc = SparkContext()
sc.addPyFile("/Users/yanyunliu/Downloads/mongo-spark-connector_2.10-2.0.0.jar")

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/users") \
    .config("spark.mongodb.output.uri", "mongodb://127.0.0.1/users") \
    .getOrCreate()


def loadDataJson(name, datafrom='', path='', Datalimit=False, DatalimitNum=0):
    # name specify the name of collection in mongodb or type of json file.
    # datafrom specify where the data come from ('mongodb' or 'json')
    # Datalimit specify if you want to use limit data. and DatalimitNum specify the number of data you want to use.
    # if use limit data, cache it after loading from mongoDB or json.

    if datafrom == 'json':
        DF = spark.read.json(path)

        print '*' * 100
        print "This is the schema in original json review file"
        print '*' * 100

        DF.printSchema()

        if Datalimit == True:
            # Use limited dataset: enable the limit and cache()
            DF = DF.limit(DatalimitNum)

    elif datafrom == 'mongodb':

        DF = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("uri",
                                                                             "mongodb://127.0.0.1/users." + name).load()

        print '*' * 100
        print "This is the schema in original mongoDB review collection"
        print '*' * 100

        DF.printSchema()

        if Datalimit == True:
            # Use limited dataset: enable the limit and cache()
            DF = DF.limit(DatalimitNum)

    return DF


if __name__ == '__main__':

    reviewpath = [
        'review', '/Users/yanyunliu/Downloads/yelp_dataset_challenge_round9/yelp_academic_dataset_review.json']
    businesspath = [
        'business', '/Users/yanyunliu/Downloads/yelp_dataset_challenge_round9/yelp_academic_dataset_business.json']
    checkinpath = [
        'checkin', '/Users/yanyunliu/Downloads/yelp_dataset_challenge_round9/yelp_academic_dataset_checkin.json']
    userpath = [
        'user', '/Users/yanyunliu/Downloads/yelp_dataset_challenge_round9/yelp_academic_dataset_user.json']
    tippath = [
        'tip', '/Users/yanyunliu/Downloads/yelp_dataset_challenge_round9/yelp_academic_dataset_tip.json']

    path_list = [reviewpath, businesspath, checkinpath, userpath, tippath]

    for path_link in path_list:

        print "Test efficiency: Loading data from mongoDB vs loading data from json file: "
        t1 = time.time()
        selectreviewDF = loadDataJson(path_link[0], 'json', path=path_link[1])
        t2 = time.time()

        print str(t2 - t1), "This is the time cost for loading data from " + path_link[0] + " json file."

        t1 = time.time()
        selectreviewDF = loadDataJson(
            path_link[0], 'mongodb', path=path_link[1])
        t2 = time.time()

        print str(t2 - t1), "This is the time cost for loading data from MongoDB " + path_link[0] + "."


sc.stop()
