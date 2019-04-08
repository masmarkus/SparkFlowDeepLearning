#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = """\n""".join(['Created by : ','Markus Paramahasti <markus@volantis.io>'])


from sparkflow.pipeline_util import PysparkPipelineWrapper
import numpy
import tensorflow as tf
import os
import requests




#url = 'https://github.com/lifeomic/sparkflow/blob/master/examples/mnist_train.csv'
#response = requests.get(url)
#with open("mnist-csv", 'wb') as f:
#    f.write(response.content)



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

spark_model = SparkAsyncDL(
    inputCol='features',
    tensorflowGraph=mg,
    tfInput='x:0',
    tfLabel='y:0',
    tfOutput='out:0',
    tfLearningRate=.001,
    iters=20,
    predictionCol='predicted',
    labelCol='labels',
    verbose=1)





if __name__ == "__main__":


    # SparkContext Initialization
    from pyspark.ml.pipeline import Pipeline
    from pyspark.context import SparkContext
    from pyspark.sql.session import SparkSession

    #sc = SparkContext('local')
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)


    #Pipeline definition
    pipe = [va, encoded, spark_model]
    #Train the model
    p = Pipeline(stages=pipe).fit(df)
    #Save the model
    p.write().overwrite().save("your/model/location")
