from conf import CONFIG


# Kafka configuration for consume message
KAFKA_SETTING_CONSUMER = {
    "bootstrap.servers": CONFIG["KAFKA_HOST"],
    "group.id": "running_group",
    "client.id": "ml-studio/pipeline-runner",
    "enable.auto.commit": False,
}

KAFKA_SETTING_PRODUCER = {
    "bootstrap.servers": CONFIG["KAFKA_HOST"],
    "group.id": "running_group",
    "client.id": "ml-studio/pipeline-runner"
}

MODUL_DICT = {
    "Transformer": "from pyspark.ml import Transformer",
    "Estimator": "from pyspark.ml import Estimator",
    "Model": "from pyspark.ml import Model",
    "Pipeline": "from pyspark.ml import Pipeline",
    "PipelineModel": "from pyspark.ml import PipelineModel",
    "Param": "from pyspark.ml.param import Param",
    "TypeConverters": "from pyspark.ml.param import TypeConverters",
    "Binarizer": "from pyspark.ml.feature import Binarizer",
    "BucketedRandomProjectionLSH": "from pyspark.ml.feature import BucketedRandomProjectionLSH",
    "BucketedRandomProjectionLSHModel": "from pyspark.ml.feature import BucketedRandomProjectionLSHModel",
    "Bucketizer": "from pyspark.ml.feature import Bucketizer",
    "ChiSqSelector": "from pyspark.ml.feature import ChiSqSelector",
    "ChiSqSelectorModel": "from pyspark.ml.feature import ChiSqSelectorModel",
    "CountVectorizer": "from pyspark.ml.feature import CountVectorizer",
    "CountVectorizerModel": "from pyspark.ml.feature import CountVectorizerModel",
    "DCT": "from pyspark.ml.feature import DCT",
    "ElementwiseProduct": "from pyspark.ml.feature import ElementwiseProduct",
    "HashingTF": "from pyspark.ml.feature import HashingTF",
    "IDF": "from pyspark.ml.feature import IDF",
    "IDFModel": "from pyspark.ml.feature import IDFModel",
    "Imputer": "from pyspark.ml.feature import Imputer",
    "ImputerModel": "from pyspark.ml.feature import ImputerModel",
    "IndexToString": "from pyspark.ml.feature import IndexToString",
    "MaxAbsScaler": "from pyspark.ml.feature import MaxAbsScaler",
    "MaxAbsScalerModel": "from pyspark.ml.feature import MaxAbsScalerModel",
    "MinHashLSH": "from pyspark.ml.feature import MinHashLSH",
    "MinHashLSHModel": "from pyspark.ml.feature import MinHashLSHModel",
    "MinMaxScaler": "from pyspark.ml.feature import MinMaxScaler",
    "MinMaxScalerModel": "from pyspark.ml.feature import MinMaxScalerModel",
    "NGram": "from pyspark.ml.feature import NGram",
    "Normalizer": "from pyspark.ml.feature import Normalizer",
    "OneHotEncoder": "from pyspark.ml.feature import OneHotEncoder",
    "PCA": "from pyspark.ml.feature import PCA",
    "PCAModel": "from pyspark.ml.feature import PCAModel",
    "PolynomialExpansion": "from pyspark.ml.feature import PolynomialExpansion",
    "QuantileDiscretizer": "from pyspark.ml.feature import QuantileDiscretizer",
    "RegexTokenizer": "from pyspark.ml.feature import RegexTokenizer",
    "RFormula": "from pyspark.ml.feature import RFormula",
    "RFormulaModel": "from pyspark.ml.feature import RFormulaModel",
    "SQLTransformer": "from pyspark.ml.feature import SQLTransformer",
    "StandardScaler": "from pyspark.ml.feature import StandardScaler",
    "StandardScalerModel": "from pyspark.ml.feature import StandardScalerModel",
    "StopWordsRemover": "from pyspark.ml.feature import StopWordsRemover",
    "StringIndexer": "from pyspark.ml.feature import StringIndexer",
    "StringIndexerModel": "from pyspark.ml.feature import StringIndexerModel",
    "Tokenizer": "from pyspark.ml.feature import Tokenizer",
    "VectorAssembler": "from pyspark.ml.feature import VectorAssembler",
    "VectorIndexer": "from pyspark.ml.feature import VectorIndexer",
    "VectorIndexerModel": "from pyspark.ml.feature import VectorIndexerModel",
    "VectorSlicer": "from pyspark.ml.feature import VectorSlicer",
    "Word2Vec": "from pyspark.ml.feature import Word2Vec",
    "Word2VecModel": "from pyspark.ml.feature import Word2VecModel",
    "LinearSVC": "from pyspark.ml.classification import LinearSVC",
    "LinearSVCModel": "from pyspark.ml.classification import LinearSVCModel",
    "LogisticRegression": "from pyspark.ml.classification import LogisticRegression",
    "LogisticRegressionModel": "from pyspark.ml.classification import LogisticRegressionModel",
    "LogisticRegressionSummary": "from pyspark.ml.classification import LogisticRegressionSummary",
    "LogisticRegressionTrainingSummary": "from pyspark.ml.classification import LogisticRegressionTrainingSummary",
    "BinaryLogisticRegressionSummary": "from pyspark.ml.classification import BinaryLogisticRegressionSummary",
    "BinaryLogisticRegressionTrainingSummary":
        "from pyspark.ml.classification import BinaryLogisticRegressionTrainingSummary",
    "DecisionTreeClassifier": "from pyspark.ml.classification import DecisionTreeClassifier",
    "DecisionTreeClassificationModel": "from pyspark.ml.classification import DecisionTreeClassificationModel",
    "GBTClassifier": "from pyspark.ml.classification import GBTClassifier",
    "GBTClassificationModel": "from pyspark.ml.classification import GBTClassificationModel",
    "RandomForestClassifier": "from pyspark.ml.classification import RandomForestClassifier",
    "RandomForestClassificationModel": "from pyspark.ml.classification import RandomForestClassificationModel",
    "NaiveBayes": "from pyspark.ml.classification import NaiveBayes",
    "NaiveBayesModel": "from pyspark.ml.classification import NaiveBayesModel",
    "MultilayerPerceptronClassifier": "from pyspark.ml.classification import MultilayerPerceptronClassifier",
    "MultilayerPerceptronClassificationModel":
        "from pyspark.ml.classification import MultilayerPerceptronClassificationModel",
    "OneVsRest": "from pyspark.ml.classification import OneVsRest",
    "OneVsRestModel": "from pyspark.ml.classification import OneVsRestModel",
    "BisectingKMeans": "from pyspark.ml.clustering import BisectingKMeans",
    "BisectingKMeansModel": "from pyspark.ml.clustering import BisectingKMeansModel",
    "BisectingKMeansSummary": "from pyspark.ml.clustering import BisectingKMeansSummary",
    "KMeans": "from pyspark.ml.clustering import KMeans",
    "KMeansModel": "from pyspark.ml.clustering import KMeansModel",
    "GaussianMixture": "from pyspark.ml.clustering import GaussianMixture",
    "GaussianMixtureModel": "from pyspark.ml.clustering import GaussianMixtureModel",
    "GaussianMixtureSummary": "from pyspark.ml.clustering import GaussianMixtureSummary",
    "LDA": "from pyspark.ml.clustering import LDA",
    "LDAModel": "from pyspark.ml.clustering import LDAModel",
    "LocalLDAModel": "from pyspark.ml.clustering import LocalLDAModel",
    "DistributedLDAModel": "from pyspark.ml.clustering import DistributedLDAModel",
    "Vector": "from pyspark.ml.linalg import Vector",
    "DenseVector": "from pyspark.ml.linalg import DenseVector",
    "SparseVector": "from pyspark.ml.linalg import SparseVector",
    "Vectors": "from pyspark.ml.linalg import Vectors",
    "Matrix": "from pyspark.ml.linalg import Matrix",
    "DenseMatrix": "from pyspark.ml.linalg import DenseMatrix",
    "SparseMatrix": "from pyspark.ml.linalg import SparseMatrix",
    "Matrices": "from pyspark.ml.linalg import Matrices",
    "ALS": "from pyspark.ml.recommendation import ALS",
    "ALSModel": "from pyspark.ml.recommendation import ALSModel",
    "AFTSurvivalRegression": "from pyspark.ml.regression import AFTSurvivalRegression",
    "AFTSurvivalRegressionModel": "from pyspark.ml.regression import AFTSurvivalRegressionModel",
    "DecisionTreeRegressor": "from pyspark.ml.regression import DecisionTreeRegressor",
    "DecisionTreeRegressionModel": "from pyspark.ml.regression import DecisionTreeRegressionModel",
    "GBTRegressor": "from pyspark.ml.regression import GBTRegressor",
    "GBTRegressionModel": "from pyspark.ml.regression import GBTRegressionModel",
    "GeneralizedLinearRegression": "from pyspark.ml.regression import GeneralizedLinearRegression",
    "GeneralizedLinearRegressionModel": "from pyspark.ml.regression import GeneralizedLinearRegressionModel",
    "GeneralizedLinearRegressionSummary": "from pyspark.ml.regression import GeneralizedLinearRegressionSummary",
    "GeneralizedLinearRegressionTrainingSummary":
        "from pyspark.ml.regression import GeneralizedLinearRegressionTrainingSummary",
    "IsotonicRegression": "from pyspark.ml.regression import IsotonicRegression",
    "IsotonicRegressionModel": "from pyspark.ml.regression import IsotonicRegressionModel",
    "LinearRegression": "from pyspark.ml.regression import LinearRegression",
    "LinearRegressionModel": "from pyspark.ml.regression import LinearRegressionModel",
    "LinearRegressionSummary": "from pyspark.ml.regression import LinearRegressionSummary",
    "LinearRegressionTrainingSummary": "from pyspark.ml.regression import LinearRegressionTrainingSummary",
    "RandomForestRegressor": "from pyspark.ml.regression import RandomForestRegressor",
    "RandomForestRegressionModel": "from pyspark.ml.regression import RandomForestRegressionModel",
    "ChiSquareTest": "from pyspark.ml.stat import ChiSquareTest",
    "Correlation": "from pyspark.ml.stat import Correlation",
    "ParamGridBuilder": "from pyspark.ml.tuning import ParamGridBuilder",
    "CrossValidator": "from pyspark.ml.tuning import CrossValidator",
    "CrossValidatorModel": "from pyspark.ml.tuning import CrossValidatorModel",
    "TrainValidationSplit": "from pyspark.ml.tuning import TrainValidationSplit",
    "TrainValidationSplitModel": "from pyspark.ml.tuning import TrainValidationSplitModel",
    "Evaluator": "from pyspark.ml.evaluation import Evaluator",
    "BinaryClassificationEvaluator": "from pyspark.ml.evaluation import BinaryClassificationEvaluator",
    "RegressionEvaluator": "from pyspark.ml.evaluation import RegressionEvaluator",
    "MulticlassClassificationEvaluator": "from pyspark.ml.evaluation import MulticlassClassificationEvaluator",
    "FPGrowth": "from pyspark.ml.fpm. import FPGrowth",
    "FPGrowthModel": "from pyspark.ml.fpm import FPGrowthModel",
    "Identifiable": "from pyspark.ml.util import Identifiable",
    "JavaMLReadable": "from pyspark.ml.util import JavaMLReadable",
    "JavaMLReader": "from pyspark.ml.util import JavaMLReader",
    "JavaMLWritable": "from pyspark.ml.util import JavaMLWritable",
    "JavaMLWriter": "from pyspark.ml.util import JavaMLWriter",
    "JavaPredictionModel": "from pyspark.ml.util import JavaPredictionModel",
    "MLReadable": "from pyspark.ml.util import MLReadable",
    "MLReader": "from pyspark.ml.util import MLReader",
    "MLWritable": "from pyspark.ml.util import MLWritable",
    "MLWriter": "from pyspark.ml.util import MLWriter",
    "custom_adapter": "from adapter import custom_adapter",
    "ClusteringEvaluator": "from pyspark.ml.evaluation import ClusteringEvaluator"
}

NUMBER_OF_TRIALS_PIPELINE = 3
NUMBER_OF_TRIALS = 3

TEMPLATE_PATH = {
    "SCRIPT": "/code_generated",
    "TEMPLATE": "/template"
}

FOOTER_PATTERN = {
    "CV": {
        "$PIPELINE_TITLE": "Cross Validator",
        "$NUMBER_OF_TRIALS_PIPELINE": str(NUMBER_OF_TRIALS_PIPELINE),
        "$SAVE_TO_HADOOP": 'from utils.hadoop.save_to_hadoop import save_model_to_hadoop'
                           '\n\t\thadoop_path_model = message["pipeline_result_path"]'
                           '\n\t\tpath = save_model_to_hadoop(model, hadoop_path_model)',
        "$STAGE_RESULT_TUPLE": "model, metric_performance, data_prediction",
        "$RETURN_TUPLE": "data_prediction, data_train, path, time_total, model, metric_performance",
        "$GET_RESULT_TUPLE": "data_prediction, data_train, path, time_total, model, metric_performance",
        "$METADATA_PARAMS": "spark, hadoop_metadata_path, PipelineType.FIT, None, data_prediction, data_train, "
                            "model, message, time_total, metric_performance",
        "$METADATA_ERROR_PARAMS": "spark, hadoop_metadata_path, PipelineType.FIT, str(er)",
    },
    "TVS": {
        "$PIPELINE_TITLE": "Train Validation Split",
        "$NUMBER_OF_TRIALS_PIPELINE": str(NUMBER_OF_TRIALS_PIPELINE),
        "$SAVE_TO_HADOOP": 'from utils.hadoop.save_to_hadoop import save_model_to_hadoop'
                           '\n\t\thadoop_path_model = message["pipeline_result_path"]'
                           '\n\t\tpath = save_model_to_hadoop(model, hadoop_path_model)',
        "$STAGE_RESULT_TUPLE": "model, metric_performance, data_prediction",
        "$RETURN_TUPLE": "data_prediction, data_train, path, time_total, model, metric_performance",
        "$GET_RESULT_TUPLE": "data_prediction, data_train, path, time_total, model, metric_performance",
        "$METADATA_PARAMS": "spark, hadoop_metadata_path, PipelineType.FIT, None, data_prediction, data_train, "
                            "model, message, time_total, metric_performance",
        "$METADATA_ERROR_PARAMS": "spark, hadoop_metadata_path, PipelineType.FIT, str(er)",
    },
    "GP": {
        "$PIPELINE_TITLE": "General Pipeline",
        "$NUMBER_OF_TRIALS_PIPELINE": str(NUMBER_OF_TRIALS_PIPELINE),
        "$SAVE_TO_HADOOP": 'from utils.hadoop.save_to_hadoop import save_model_to_hadoop'
                           '\n\t\thadoop_path_model = message["pipeline_result_path"]'
                           '\n\t\tpath = save_model_to_hadoop(model, hadoop_path_model)',
        "$STAGE_RESULT_TUPLE": "model, data_prediction",
        "$RETURN_TUPLE": "model, data_prediction, data_train, path, time_total",
        "$GET_RESULT_TUPLE": "model, data_prediction, data_train, path, time_total",
        "$METADATA_PARAMS": "spark, hadoop_metadata_path, PipelineType.FIT, None, data_prediction, data_train, "
                            "model, message, time_total",
        "$METADATA_ERROR_PARAMS": "spark, hadoop_metadata_path, PipelineType.FIT, str(er)",
    },
    "TRANSFORM": {
        "$PIPELINE_TITLE": "Data Transformation",
        "$NUMBER_OF_TRIALS_PIPELINE": str(NUMBER_OF_TRIALS_PIPELINE),
        "$SAVE_TO_HADOOP": 'from utils.hadoop.save_to_hadoop import save_data_to_hadoop'
                           '\n\t\tfrom modules.extend_dataframe import create_new_data'
                           '\n\n\t\thadoop_path_data = message["pipeline_result_path"]'
                           '\n\t\tinput_col = message["data"]["column_input_dictionary"]'
                           '\n\t\tnew_data = create_new_data(data_prediction, input_col, spark)'
                           '\n\t\tpath, data_payload = save_data_to_hadoop(message, new_data, hadoop_path_data)',
        "$STAGE_RESULT_TUPLE": "model, data_prediction",
        "$RETURN_TUPLE": "new_data, data_train, path, time_total, data_payload",
        "$GET_RESULT_TUPLE": "data_prediction, data_train, path, time_total, data_payload",
        "$METADATA_PARAMS": "spark, hadoop_metadata_path, PipelineType.TRANSFORM, None, data_prediction, data_train, "
                            "None, message, time_total, None, data_payload",
        "$METADATA_ERROR_PARAMS": "spark, hadoop_metadata_path, PipelineType.TRANSFORM, str(er)",
    }
}

# Hadoop path
HADOOP_PATH = {
    "DATA": "/ml_studio/data",
    "MODEL": "/ml_studio/model",
    "METADATA": "/ml_studio/metadata",
    "SCRIPT": "/ml_studio/script",
    "PACKAGE": "/ml_studio/package"
}

# Message version of Pipeline runner
MESSAGE_VERSION = "1.1.2"

PACKAGE_PATH = '/packages'

# Wrapped all settings
SETTING = {
    "KAFKA_SETTING_CONSUMER": KAFKA_SETTING_CONSUMER,
    "KAFKA_SETTING_PRODUCER": KAFKA_SETTING_PRODUCER,
    "MODUL_DICT": MODUL_DICT,
}
