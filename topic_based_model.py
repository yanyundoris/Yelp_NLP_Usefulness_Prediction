#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import nltk
import time

'''
# Set the path for spark installation
# this is the path where you have built spark using sbt/sbt assembly
os.environ['SPARK_HOME'] = "/Applications/spark-2.1.0"

# Append to PYTHONPATH so that pyspark could be found
sys.path.append("/Applications/spark-2.1.0/python")

os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--packages org.mongodb.spark:mongo-spark-connector_2.10:2.0.0 --driver-memory 5g  --executor-memory 5g pyspark-shell")

'''
# Now we are ready to import Spark Modules
try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.ml import Pipeline
    from pyspark.ml.linalg import Vectors, VectorUDT
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import *
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
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
'''
sc = SparkContext()
sc.addPyFile("/Users/yanyunliu/Downloads/mongo-spark-connector_2.10-2.0.0.jar")

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/users") \
    .config("spark.mongodb.output.uri", "mongodb://127.0.0.1/users") \
    .getOrCreate()
'''

"""
Load Review data and read it into a Dataframe named as reviewDF, select useful columns and save it as selectreviewDF
"""


def UsefulnessPredictionLDA(trainingdata, model):
    # Data Preprocessing
    tokenizer = Tokenizer(inputCol="review_text", outputCol="tokens_word")

    remover = StopWordsRemover(
        inputCol="tokens_word", outputCol="filtered_tokens_word")
    cv = CountVectorizer(inputCol="filtered_tokens_word",
                         outputCol="raw_features", minDF=2.0)
    idf = IDF(inputCol="raw_features", outputCol="features")

    # Extract LDA topic feature
    lda = LDA(k=30, maxIter=10)
    if model == 'RandomForest':
        model = RandomForestRegressor(featuresCol="topicDistribution")
    pipeline = Pipeline(stages=[tokenizer, remover, cv, idf, lda, model])
    evaluator_rmse = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    paramGrid = ParamGridBuilder() \
        .addGrid(cv.vocabSize, [150, 200, 250]) \
        .build()
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator_rmse,
                              numFolds=4)  # use 3+ folds in practice
    cvModel = crossval.fit(trainingdata)
    # Explain params for the selected model
    print cvModel.explainParams()
    return cvModel


def UsefulnessPredictionLDAWithoutCV(trainingdata, model):
    # Data Preprocessing
    tokenizer = Tokenizer(inputCol="review_text", outputCol="tokens_word")
    remover = StopWordsRemover(
        inputCol="tokens_word", outputCol="filtered_tokens_word")
    cv = CountVectorizer(inputCol="filtered_tokens_word",
                         outputCol="raw_features", minDF=2.0, vocabSize=250)
    idf = IDF(inputCol="raw_features", outputCol="features")

    # Extract LDA topic feature
    lda = LDA(k=30, maxIter=10)
    if model == 'RandomForest':
        model = RandomForestRegressor(featuresCol="topicDistribution")

    pipeline = Pipeline(stages=[tokenizer, remover, cv, idf, lda, model])
    evaluator_rmse = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")

    cvModel = pipeline.fit(trainingdata)

    # Explain params for the selected model
    print cvModel.explainParams()
    return cvModel


def GetPredictionError(cvModel, testData, evaluation_method, col_name):
    predictions = cvModel.transform(testData)
    print predictions.count()

    evaluator_rmse = RegressionEvaluator(
        labelCol="label", predictionCol=col_name, metricName=evaluation_method)  # "rmse"
    rmse = evaluator_rmse.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    return predictions, rmse


def GetRecomList(predictions, partition_by, order_by, rank_num):
    # predictions is a dataframe which contain id and predict result(score).
    # partition_by is a name of column in predictions which identify the key to partition by
    # order_by is a name of column in predictions which identify the score we need to sort
    # rank_num specify how many records you want to return for each partition
    window = Window \
        .partitionBy(predictions[partition_by])\
        .orderBy(predictions[order_by].desc())
    recomlistdDF = predictions.select(partition_by, order_by, rank().over(window).alias('rank')) \
                              .filter(col('rank') <= rank_num)
    print "Get num of review list: ", recomlistdDF.count()
    return recomlistdDF


def GetBaselineModelError(trainingData, testData, evaluation_method, col_name):
    baseline_globalavg = trainingData.select('label').agg(
        {"label": "avg"}).collect()[0]['avg(label)']
    testData = testData.select(
        '*', lit(float(baseline_globalavg)).alias(col_name))
    print "The global average for usefulness on training data is", baseline_globalavg

    evaluator_rmse = RegressionEvaluator(
        labelCol="label", predictionCol=col_name, metricName=evaluation_method)  # "rmse"
    baseline_rmse = evaluator_rmse.evaluate(testData)
    print "Root Mean Squared Error (RMSE) for baseline model is", baseline_rmse
    return testData, baseline_rmse


if __name__ == '__main__':
    reviewpath = '/Users/yanyunliu/Downloads/yelp_dataset_challenge_round9/yelp_academic_dataset_review.json'
    reviewpath_light = '/Users/yanyunliu/Downloads/yelp_training_set/yelp_training_set_review.json'

    t1 = time.time()
    selectreviewDF = loadDataJson(
        'review', datafrom='json', path=reviewpath, Datalimit=True, DatalimitNum=5000)

    selectreviewDF = selectreviewDF.select(selectreviewDF['review_id'], selectreviewDF['business_id'], selectreviewDF['user_id'],
                                           selectreviewDF['text'], selectreviewDF['useful']) \
        .withColumnRenamed('useful', 'label') \
        .withColumnRenamed('text', 'review_text')

    selectreviewDF.cache()

    t2 = time.time()

    print str(t2 - t1), "This is the time cost for loading data from MongoDB."
    print 'num of rows in Dataframe selectreviewDF:', "*" * 100
    print selectreviewDF.count()

    (trainingData, testData) = selectreviewDF.randomSplit([0.7, 0.3], seed=111)

    # trainingData.show()
    print 'num of rows in Dataframe trainingData:', "*" * 100
    print trainingData.count()
    print 'num of rows in Dataframe testData:', "*" * 100
    # testData.show()
    print testData.count()

    testData, baseline_rmse = GetBaselineModelError(
        trainingData, testData, "rmse", "baseline_prediction")
    nModel = UsefulnessPredictionLDAWithoutCV(trainingData, 'RandomForest')
    predictions, rmse = GetPredictionError(
        nModel, testData, "rmse", "prediction")
    print 'num of rows in Dataframe predictions:', "*" * 100
    print predictions.count()

    recomlistdDF = GetRecomList(predictions, 'business_id', 'prediction', 1)
    recomlistdDF.show()

    sc.stop()
