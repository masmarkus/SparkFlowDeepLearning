from os import getenv
from os.path import abspath

# Hadoop
HADOOP_CONF_DIR = getenv("HADOOP_CONF_DIR", "{0}/tests/test_main/hadoop_conf".format(abspath("")))

# Kafka hosts
KAFKA_HOST = getenv("KAFKA_HOST", "0.0.0.0:29092")

# Topic kafka for sending result of model or data
KAFKA_PIPELINE_INITIAL_TOPIC = getenv("KAFKA_PIPELINE_INITIAL_TOPIC", "pipeline_initial_runner")
KAFKA_PIPELINE_RESULT_TOPIC = getenv("KAFKA_PIPELINE_RESULT_TOPIC", "pipeline_result_runner")

# Spark executable code depedencies
SPARK_CODE_DEPEDENCIES_URL = getenv("SPARK_CODE_DEPEDENCIES_URL",
                                    "hdfs:///ml_studio/package/modules.zip")

# Spark cluster configuration
CLIENT_SPARK_VERSION = getenv("CLIENT_SPARK_VERSION", "2.3.2")
SPARK_EXECUTOR_MEMORY = getenv("SPARK_EXECUTOR_MEMORY", "1g")
SPARK_EXECUTOR_CORES = getenv("SPARK_EXECUTOR_CORES", "1")
SPARK_CORES_MAX = getenv("SPARK_CORES_MAX", "1")
SPARK_MASTER = getenv("SPARK_MASTER", "mesos://dispatcher.spark.marathon.l4lb.thisdcos.directory:7077")
SPARK_MESOS_EXECUTOR_DOCKER_IMAGE = getenv("SPARK_MESOS_EXECUTOR_DOCKER_IMAGE",
                                           "tmunir/spark:2.7.0-2.4.0-hadoop-2.7")
SPARK_MESOS_URIS = getenv("SPARK_MESOS_URIS",
                          "http://api.hdfs.marathon.l4lb.thisdcos.directory/v1/endpoints/hdfs-site.xml,"
                          "http://api.hdfs.marathon.l4lb.thisdcos.directory/v1/endpoints/core-site.xml")
SPARK_REST_API = getenv("SPARK_REST_API", "http://178.128.210.50:13502")
SPARK_JOB_MAX_RUNTIME = float(getenv("SPARK_JOB_MAX_RUNTIME", "1"))
SHOW_STATUS_TIME = int(getenv("SHOW_STATUS_TIME", "10"))

# Running time limit on spark cluster
MAX_RUNNING_TIME_HOUR = float(getenv("MAX_RUNNING_TIME_HOUR", "1"))

# Wrapped all env to dictionary
CONFIG = {
    "KAFKA_HOST": KAFKA_HOST,
    "KAFKA_PIPELINE_INITIAL_TOPIC": KAFKA_PIPELINE_INITIAL_TOPIC,
    "KAFKA_PIPELINE_RESULT_TOPIC": KAFKA_PIPELINE_RESULT_TOPIC,
    "SPARK_CODE_DEPEDENCIES_URL": SPARK_CODE_DEPEDENCIES_URL,
    "CLIENT_SPARK_VERSION": CLIENT_SPARK_VERSION,
    "SPARK_EXECUTOR_MEMORY": SPARK_EXECUTOR_MEMORY,
    "SPARK_EXECUTOR_CORES": SPARK_EXECUTOR_CORES,
    "SPARK_CORES_MAX": SPARK_CORES_MAX,
    "SPARK_MASTER": SPARK_MASTER,
    "SPARK_MESOS_EXECUTOR_DOCKER_IMAGE": SPARK_MESOS_EXECUTOR_DOCKER_IMAGE,
    "SPARK_MESOS_URIS": SPARK_MESOS_URIS,
    "SPARK_REST_API": SPARK_REST_API,
    "SPARK_JOB_MAX_RUNTIME": SPARK_JOB_MAX_RUNTIME,
    "SHOW_STATUS_TIME": SHOW_STATUS_TIME,
    "MAX_RUNNING_TIME_HOUR": MAX_RUNNING_TIME_HOUR,
}
