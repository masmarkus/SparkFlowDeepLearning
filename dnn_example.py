#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = """\n""".join(['Created by : ','Markus Paramahasti <markus@volantis.io>'])

from sparkflow.pipeline_util import PysparkPipelineWrapper
from sparkflow.graph_utils import build_graph
from sparkflow.tensorflow_async import SparkAsyncDL
from pyspark.ml.feature import VectorAssembler, OneHotEncoder

import numpy
import tensorflow as tf
import os
import requests




#url = 'https://github.com/lifeomic/sparkflow/blob/master/examples/mnist_train.csv'
#response = requests.get(url)
#with open("mnist-csv", 'wb') as f:
#    f.write(response.content)


# SparkContext Initialization
from pyspark.ml.pipeline import Pipeline
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession


#sc = SparkContext('local')
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)


#dataset preparation
df = spark.read.option("inferSchema", "true").csv('mnist_train.csv')
#Selecting features from dataset
va = VectorAssembler(inputCols=df.columns[1:785], outputCol='features')
#Selecting labels from dataset
encoded = OneHotEncoder(inputCol='_c0', outputCol='labels', dropLast=False)


# Graph definition
def small_model():
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y = tf.placeholder(tf.float32, shape=[None, 10], name='y')
    layer1 = tf.layers.dense(x, 256, activation=tf.nn.relu)
    layer2 = tf.layers.dense(layer1, 256, activation=tf.nn.relu)
    out = tf.layers.dense(layer2, 10)
    z = tf.argmax(out, 1, name='out')
    loss = tf.losses.softmax_cross_entropy(y, out)
    return loss


# Build the graph
mg = build_graph(small_model)

# Train Parameter
spark_model = SparkAsyncDL(
    inputCol='features',
    tensorflowGraph=mg,
    tfInput='x:0',
    tfLabel='y:0',
    tfOutput='out:0',
    tfLearningRate=.001,
    iters=10,
    predictionCol='predicted',
    labelCol='labels',
    verbose=1)



if __name__ == "__main__":

    #Pipeline definition
    #pipe = [va, encoded, spark_model]

    #Train the model
    try:
        import time
        start_time = time.time()
        p = Pipeline(stages=[va, encoded, spark_model]).fit(df)
        print("--- %s seconds ---" % (time.time() - start_time))

        p.write().save("dnn_model")
    except Exception as e:
        print ("Error --> ", e)

    #Save the model
    #p.write().overwrite().save("dnn_model")


    #exec(open("dnn_model.py").read())
