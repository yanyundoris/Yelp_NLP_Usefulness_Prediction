# Yelp_NLP_Usefulness_Prediction
This is a natural language processing project using Pyspark and NLTK. The completed version is in firiceguo/Recommendation-NLP, which is a group project of HKUST 5003 Big data computing, and this repository contains NLP part only.


# Function Set

There are three files in final version: 

- `Sentiment_based_model.py`

- `topic_based_model.py`

- `share.py`

## `Sentiment_based_model.py`

Reminder: You need to install `nltk` package before you run it. 

Besides, some corpuses are required as well. You may need to use `nltk.download()` to download them:

1. SentimentIntensityAnalyzer

2. Tokenizer

### Function list:

1. `loadData`: Load data from json or mongoDB

2. `RemovePunct`：Remove punctuation from raw text

3. `RemoveEmpty`：remove empty string from a list.

4. `GetCharacter_List`：Get Part of Speech Tagging Result

5. `GetSentimentScore`：Get Sentiment Scores

6. `SentimentFeatureEngineer`：Combine all sentiment-related feature together

7. `GetBaselineModelError`：Calculate baseline and RMSE

8. `UsefulnessPredictionSentment`：Prediction with sentiment based model including Cross Validation

9. `UsefulnessPredictionSentmentWithoutCV`: Prediction with sentiment based model  without Cross Validation

10. `GetPredictionError`：Calculate RMSE for predicted result

11. `GetRecomList`：Return the final recommendation list

## `topic_based_model.py`

### Function list：

1. `loadData`：Load data from json or mongoDB（overlapping with `Sentiment_based_model.py`）

2. `UsefulnessPredictionLDA`：Prediction with topic based model including Cross Validation

3. `UsefulnessPredictionLDAWithoutCV`：Prediction with topic based model without Cross Validation

4. `GetPredictionError`：Calculate RMSE for predicted result（overlapping with `Sentiment_based_model.py`）

5. `GetRecomList`：Return the final recommendation list（overlapping with `Sentiment_based_model.py`）

6. `GetBaselineModelError`：Calculate baseline and RMSE（overlapping with `Sentiment_based_model.py`）

Totally `13` functions, `loadData()` and `GetRecomList()` will be shared to all of us, so I put them in share.py

## How to use it:

`Topic_based_ model.py`

```python
selectreviewDF = loadDataJson('review', datafrom='mongodb', Datalimit=True, DatalimitNum=5000)
selectreviewDF = selectreviewDF.select(selectreviewDF['review_id'], selectreviewDF['business_id'], selectreviewDF['user_id'],
                                       selectreviewDF['text'], selectreviewDF['useful']) \
    .withColumnRenamed('useful', 'label') \
    .withColumnRenamed('text', 'review_text') #select waht column you will use
selectreviewDF.cache() #cache DF if using limited dataset.
(trainingData, testData) = selectreviewDF.randomSplit([0.7, 0.3],seed=111)

testData, baseline_rmse = GetBaselineModelError(trainingData, testData, "rmse","baseline_prediction")

nModel = UsefulnessPredictionLDAWithoutCV(trainingData,'RandomForest')

predictions, rmse = GetPredictionError(nModel,testData,"rmse","prediction")

recomlistdDF = GetRecomList(predictions,'business_id','prediction',1)

recomlistdDF.show()
```

`Sentiment_based_model.py`

```python
selectreviewDF = loadDataJson('review', datafrom='mongodb',Datalimit=True, DatalimitNum=1000)
#selectreviewDF = loadReviewDataJson('json', review_path=reviewpath, Datalimit=True, DatalimitNum=50)
selectreviewDF = selectreviewDF.select(selectreviewDF['review_id'], selectreviewDF['business_id'], selectreviewDF['user_id'],
                                       selectreviewDF['text'], selectreviewDF['useful']) \
                               .withColumnRenamed('useful', 'label') \
                               .withColumnRenamed('text', 'review_text')

# if you use limit data it must be cached.
selectreviewDF.cache()
selectreviewDF = SentimentFeatureEngineer(selectreviewDF)
(trainingData, testData) = selectreviewDF.randomSplit([0.7, 0.3])
testData, baseline_rmse = GetBaselineModelError(trainingData, testData, 'rmse', 'baseline_prediction')

# cvMddel = UsefulnessPredictionSentment(trainingData, 'RandomForest')
nMode = UsefulnessPredictionSentmentWithoutCV(trainingData, 'RandomForest')

# predictions, rmse = GetPredictionError(cvMddel,testData,'rmse', 'prediction')
predictions, rmse = GetPredictionError(nMode, testData, 'rmse', 'prediction')

recomlistdDF = GetRecomList(predictions, 'business_id', 'prediction', 1)
recomlistdDF.filter(recomlistdDF['label']>0).show()
```
  
## share.py

`loadData()`
  
You need to Prepare spark & mongoDB environment:

```python
import os
import sys
os.environ['SPARK_HOME'] = "/Applications/spark-2.1.0"
sys.path.append("/Applications/spark-2.1.0/python")
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--packages org.mongodb.spark:mongo-spark-connector_2.10:2.0.0 --driver-memory 5g  --executor-memory 5g pyspark-shell")
try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.sql import SparkSession
except ImportError as e:
    print ("Error importing Spark Modules", e)
    sys.exit(1)

sc = SparkContext()
sc.addPyFile("/Users/yanyunliu/Downloads/mongo-spark-connector_2.10-2.0.0.jar")
spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/users") \
    .config("spark.mongodb.output.uri", "mongodb://127.0.0.1/users") \
    .getOrCreate()
```

    
`GetRecomList()`

You need to import

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import rank
```

You can find comments in `share.py`, where I explain how to use these two functions.




