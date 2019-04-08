
# SparkFlow

This is an implementation of Tensorflow on Spark. The goal of this library is to provide a simple, understandable interface in using Tensorflow on Spark. With SparkFlow, you can easily integrate your deep learning model with a ML Spark Pipeline. Underneath, SparkFlow uses a parameter server to train the Tensorflow network in a distributed manner. Through the api, the user can specify the style of training, whether that is Hogwild or async with locking. [GitHub](https://github.com/lifeomic/sparkflow)

## Why should I use this?
While there are other libraries that use Tensorflow on Apache Spark, Sparkflow's objective is to work seemlessly
with ML Pipelines, provide a simple interface for training Tensorflow graphs, and give basic abstractions for
faster development. For training, Sparkflow uses a parameter server which lives on the driver and allows for asynchronous training. This tool
provides faster training time when using big data.

## Installation

Install sparkflow via pip: `$ pip install sparkflow`
SparkFlow requires Apache Spark >= 2.0, flask, and Tensorflow to all be installed.
