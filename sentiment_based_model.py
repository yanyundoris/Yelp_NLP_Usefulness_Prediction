#!/usr/bin/python2
# -*- coding: utf-8 -*-

import sys
import nltk
import time
import re
from collections import Counter
import string


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
    from pyspark.sql.types import IntegerType, FloatType, StringType, ArrayType
    from pyspark.sql.window import Window
    from pyspark.sql.functions import rank, col
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.feature import Normalizer
    from pyspark.ml.linalg import Vectors, VectorUDT
    from pyspark.sql.functions import udf
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer


except ImportError as e:
    print("Error importing Spark Modules", e)
    sys.exit(1)


def RemovePunct(subset):
    # Replace punctuation from raw text
    for s in string.punctuation:
        subset = subset.replace(s, "")
    # Replace digit from raw text otherwise it will return error when applying
    # character tagging.
    number_pattern = re.compile("\d")
    subset = number_pattern.sub(" ", subset)

    return subset


def RemoveEmpty(line):
    # Replace emply entry from a list of words.
    new_line = []
    for item in line:
        if item != '':
            new_line.append(item)
    return new_line


def GetCharacter_List(line):
    # Tagging words by nltk
    adj_count = 0
    noun_count = 0
    verb_count = 0
    adv_count = 0

    if len(line) != 0:
        tag_dict = nltk.pos_tag(line)
        tag_dict = dict(tag_dict)

        cc = Counter()
        for key, value in tag_dict.items():
            cc[value] += 1

        for key in cc.keys():
            if key.startswith('JJ'):
                adj_count = adj_count + cc[key]
            elif key.startswith('NN'):
                noun_count = adj_count + cc[key]
            elif key.startswith('VB'):
                verb_count = verb_count + cc[key]
            elif key.startswith('RB'):
                adv_count = adv_count + cc[key]
    chara_list = [adj_count, noun_count, verb_count, adv_count]
    return chara_list


def GetSentimentScore(line):
    # Get sentiment score within four perspectives: negative, positive, neutral and compound
    # Merge a list(str of word) to str of line
    line = ' '.join(line)

    sid = SentimentIntensityAnalyzer()
    score = sid.polarity_scores(line)

    sentiment_score = [int(score['neg'] * 100), int(score['neu'] * 100),
                       int(score['pos'] * 100), int(score['compound'] * 100)]

    return sentiment_score


def SentimentFeatureEngineer(selectreviewDF):
    RemovePunct_udf = udf(RemovePunct, StringType())
    countTokens_udf = udf(lambda words: len(words), IntegerType())
    RemoveEmptyEntry_udf = udf(RemoveEmpty, ArrayType(StringType()))
    GetCharacter_List_udf = udf(GetCharacter_List, ArrayType(IntegerType()))
    GetSentimentScore_udf = udf(GetSentimentScore, ArrayType(IntegerType()))
    list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())

    selectreviewDF = selectreviewDF.withColumn(
        'remove_punc', RemovePunct_udf(selectreviewDF['review_text']))

    tokenizer = Tokenizer(inputCol="remove_punc", outputCol="tokens_word")
    selectreviewDF = tokenizer.transform(selectreviewDF)

    # Remender: Do not combine the structure here otherwise you will get an
    # error. (some columns will not be found)

    selectreviewDF = selectreviewDF.withColumn(
        'num', countTokens_udf(selectreviewDF['tokens_word']))
    selectreviewDF = selectreviewDF.withColumn(
        'filtered_review_text_new', RemoveEmptyEntry_udf(selectreviewDF['tokens_word']))

    selectreviewDF = selectreviewDF.withColumn('Character_adj',
                                               GetCharacter_List_udf(selectreviewDF['filtered_review_text_new'])[0]) \
        .withColumn('Character_noun', GetCharacter_List_udf(selectreviewDF['filtered_review_text_new'])[1]) \
        .withColumn('Character_verb', GetCharacter_List_udf(selectreviewDF['filtered_review_text_new'])[2]) \
        .withColumn('Character_adv', GetCharacter_List_udf(selectreviewDF['filtered_review_text_new'])[3]) \
        .withColumn('sentiment_neg', GetSentimentScore_udf(selectreviewDF['filtered_review_text_new'])[0]) \
        .withColumn('sentiment_neu', GetSentimentScore_udf(selectreviewDF['filtered_review_text_new'])[1]) \
        .withColumn('sentiment_pos', GetSentimentScore_udf(selectreviewDF['filtered_review_text_new'])[2]) \
        .withColumn('sentiment_compound', GetSentimentScore_udf(selectreviewDF['filtered_review_text_new'])[3])

    return selectreviewDF


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


def UsefulnessPredictionSentment(trainingdata, model):
    # Data Preprocessing
    assembler = VectorAssembler(
        inputCols=['num', 'sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound',
                   'Character_adj', 'Character_noun', 'Character_verb', 'Character_adv'],
        outputCol="features")

    featureIndexer = VectorIndexer(
        inputCol="features", outputCol="indexedFeatures")

    if model == 'RandomForest':
        model = RandomForestRegressor(featuresCol="indexedFeatures")

    pipeline = Pipeline(stages=[assembler, featureIndexer, model])

    evaluator_rmse = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")

    paramGrid = ParamGridBuilder() \
        .addGrid(featureIndexer.maxCategories, [3, 4, 5]) \
        .build()

    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator_rmse,
                              numFolds=2)  # use 3+ folds in practice

    cvModel = crossval.fit(trainingdata)

    # Explain params for the selected model
    print cvModel.explainParams()

    return cvModel


def UsefulnessPredictionSentmentWithoutCV(trainingdata, model):
    # Data Preprocessing
    assembler = VectorAssembler(
        inputCols=['num', 'sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound', 'Character_adj',
                   'Character_noun', 'Character_verb', 'Character_adv'],
        outputCol="features")

    featureIndexer = VectorIndexer(
        inputCol="features", outputCol="indexedFeatures", maxCategories=4)

    if model == 'RandomForest':
        model = RandomForestRegressor(featuresCol="indexedFeatures")

    pipeline = Pipeline(stages=[assembler, featureIndexer, model])

    evaluator_rmse = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")

    Model = pipeline.fit(trainingdata)

    return Model


def GetPredictionError(Model, testData, evaluation_method, col_name):
    predictions = Model.transform(testData)
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

    recomlistdDF = predictions.select('review_id', partition_by, order_by, 'label', rank().over(window).alias('rank')) \
                              .filter(col('rank') <= rank_num)

    print "Get num of review list: ", recomlistdDF.count()

    return recomlistdDF


if __name__ == '__main__':
    reviewpath = '/PATH_OF_YELP/yelp_academic_dataset_review.json'
    reviewpath_light = '/PATH_OF_LIGHT_DATA/yelp_training_set_review.json'

    t1 = time.time()

    selectreviewDF = loadDataJson(
        'review', datafrom='mongodb', Datalimit=True, DatalimitNum=100000)
    #selectreviewDF = loadReviewDataJson('json', review_path=reviewpath, Datalimit=True, DatalimitNum=50)

    selectreviewDF = selectreviewDF.select(selectreviewDF['review_id'], selectreviewDF['business_id'], selectreviewDF['user_id'],
                                           selectreviewDF['text'], selectreviewDF['useful']) \
        .withColumnRenamed('useful', 'label') \
        .withColumnRenamed('text', 'review_text')

    print selectreviewDF.count()

    # if you use limit data it must be cached.
    selectreviewDF.cache()

    selectreviewDF = SentimentFeatureEngineer(selectreviewDF)
    # print selectreviewDF.count()

    selectreviewDF.cache()

    (trainingData, testData) = selectreviewDF.randomSplit([0.7, 0.3])
    testData, baseline_rmse = GetBaselineModelError(
        trainingData, testData, 'rmse', 'baseline_prediction')
    # print testData.count()

    #cvMddel = UsefulnessPredictionSentment(trainingData, 'RandomForest')
    nMode = UsefulnessPredictionSentmentWithoutCV(trainingData, 'RandomForest')

    #predictions, rmse = GetPredictionError(cvMddel,testData,'rmse', 'prediction')
    predictions, rmse = GetPredictionError(
        nMode, testData, 'rmse', 'prediction')

    recomlistdDF = GetRecomList(predictions, 'business_id', 'prediction', 1)
    recomlistdDF.show()
    t2 = time.time()
    print "Time Cost Totally: ", str(t2 - t1), "Minutes: ", str((t2 - t1) / 60),  "Seconds: ", str((t2 - t1) % 60)

    sc.stop()
