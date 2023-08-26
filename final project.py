#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics
import matplotlib.pyplot as plt

# Create a Spark session
spark = SparkSession.builder.appName("PhishingDetection").getOrCreate()

# Load the phishing dataset from your distributed file system
dataset_path = "s3://varun2207/dataset_phishing.csv"
df = spark.read.csv(dataset_path, header=True, inferSchema=True)

# Data preprocessing with Spark ML
label_indexer = StringIndexer(inputCol="status", outputCol="label")
feature_columns = [col for col in df.columns if col != "status"]
feature_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Split dataset into training and testing sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Define a Linear Regression model
lr = LinearRegression(featuresCol="scaled_features", labelCol="label")

# Define a pipeline with stages for preprocessing and model training
pipeline_lr = Pipeline(stages=[label_indexer, feature_assembler, scaler, lr])

# Train the Linear Regression model
model_lr = pipeline_lr.fit(train_data)

# Make predictions
predictions_lr = model_lr.transform(test_data)

# Evaluate the model using RegressionMetrics
prediction_and_label_lr = predictions_lr.select("prediction", "label").rdd
metrics_lr = RegressionMetrics(prediction_and_label_lr)
rmse_lr = metrics_lr.rootMeanSquaredError
r2_lr = metrics_lr.r2

# Print evaluation metrics
print("Linear Regression RMSE:", rmse_lr)
print("Linear Regression R2:", r2_lr)

# Stop the Spark session
spark.stop()

