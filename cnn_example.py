

import tensorflow as tf


#Spark Initialization
spark = SparkSession.builder \
    .appName("examples") \
    .master('local[8]').config('spark.driver.memory', '2g') \
    .getOrCreate()

# Dataset preparation
df = spark.read.option("inferSchema", "true").csv('mnist_train.csv').orderBy(rand())
# Features selection
va = VectorAssembler(inputCols=df.columns[1:785], outputCol='features')
# Labels encoding
encoded = OneHotEncoder(inputCol='_c0', outputCol='labels', dropLast=False)

# CNN Graph definition
def cnn_model():
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y = tf.placeholder(tf.float32, shape=[None, 10], name='y')
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
    conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
    conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
    conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
    fc1 = tf.layers.flatten(conv2)
    out = tf.layers.dense(fc1, 10)
    z = tf.argmax(out, 1, name='out')
    loss = tf.losses.softmax_cross_entropy(y, out)
    return loss

# Build the graph
mg = build_graph(cnn_model)

spark_model = SparkAsyncDL(
        inputCol='features',
        tensorflowGraph=mg,
        tfInput='x:0',
        tfLabel='y:0',
        tfOptimizer='adam',
        miniBatchSize=300,
        miniStochasticIters=-1,
        shufflePerIter=True,
        iters=20,
        tfLearningRate=.0001,
        predictionCol='predicted',
        labelCol='labels',
        verbose=1
    )


if __name__ == '__main__':

    from pyspark.ml.pipeline import Pipeline

    # Pipeline definition
    pipe = [va, encoded, spark_model]
    # Train the CNN model
    p = Pipeline(stages=pipe).fit(df)
    # Save the model
    p.save("cnn_model")
