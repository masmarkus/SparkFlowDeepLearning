# Import base
import timeit
import traceback, sys
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext

# Intialize spark context
conf = SparkConf()
sc = SparkContext(conf=conf)
sc.addPyFile("hdfs:///ml_studio/package/modules.zip")

# Import from custom depedencies
from modules.metadata import Metadata
from modules.metadata import PipelineType
from utils.joiner import get_and_join_data
from utils.logger import custom_logger

# Initialize main variables
spark = SparkSession(sc).builder.getOrCreate()
logger = custom_logger()
pipeline = {'user_id': 'a878b137-d9c7-4217-9b91-f09b7837d13d', 'date': '2019-02-22T03:39:14.530Z', 'username': '', 'pipeline_group_name': 'HPP with LR GP Testing 14042019', 'composer_id': '79b09cd8-3653-11e9-be8a-0242ac110002', 'parents_data_id': ['7c4670cf-0f49-4091-93b0-945ce8c67b54'], 'parents_user_id_model': [], 'parents_model_id': [], 'parents_user_id_data': ['a878b137-d9c7-4217-9b91-f09b7837d13d'], 'parents_data_price': [0],
'parents_model_price': [], 'numFolds': 3, 'trainRatio': 1, 'last_stage': 'fit', 'type': 'GeneralPipeline', 'output_col': 'all', 'estimatorParamMaps': {'ParamGridBuilder': [{'LinearRegression': 'None'}]}, 'evaluator': {'name': 'MulticlassClassificationEvaluator', 'params': {'metricName': 'f1'}}, 'estimator': [{'LinearRegression': {'labelCol': 'price', 'featuresCol': '1_vector', 'predictionCol': 'prediction', 'aggregationDepth': 2, 'solver': 'auto', 'standardization': True, 'fitIntercept': True, 'elasticNetParam': 0, 'maxIter': 100, 'regParam': 0, 'tol': 1e-06, 'loss': 'squaredError', 'epsilon': 1.35}}], 'data': {'path': ['file:////Users/tmunir/Hobby/Volantis/Galaxy/hadoop/ml_studio/source_dataset/7c4670cf-0f49-4091-93b0-945ce8c67b54/1/1546403372058'], 'action': {'adapter': [[{'from': [{'value': 'n818134795', 'datatype': 'string'}, {'value': 'n1624556946', 'datatype': 'string'}, {'value': 'p1433103548', 'datatype': 'string'}, {'value': 'n247383290', 'datatype': 'string'}, {'value': 'n1552720724', 'datatype': 'string'}], 'to': [{'value': '1_vector', 'datatype': 'vector'}]}, {'from': [{'value': 'p77381929', 'datatype': 'string'}], 'to': [{'value': 'price', 'datatype': 'double'}]}]], 'join': 'None'}, 'hash_map': [['n818134795', 'p77381929', 'n1624556946', 'p1433103548', 'n247383290', 'n1552720724']], 'hash_dict': {'n818134795': {'name': 'size', 'datatype': 'double'}, 'p77381929': {'name': 'price', 'datatype': 'double'}, 'n1624556946': {'name': 'bathroom', 'datatype': 'int'}, 'p1433103548': {'name': 'bedroom', 'datatype': 'int'}, 'n247383290': {'name': 'garage_cars', 'datatype': 'int'}, 'n1552720724': {'name': 'land_size', 'datatype': 'int'}}, 'input_col': ['1_vector', 'price'], 'column_input_dictionary': {'1_vector': {'datatype': 'vector', 'columns': ['size', 'bathroom', 'bedroom', 'garage_cars', 'land_size'], 'columns_datatype': ['double', 'int', 'int', 'int', 'int']}, 'price': {'datatype': 'double'}}}, 'fit_id': '79b09ab2-3653-11e9-be8a-0242ac110002', 'access_token': 'EZ66yi6GZWbDOcvoooXajBWbCY3GrKcQwRRq83egVztRPDa6zg6z7is9zrbR3yoq', 'id': 'b434073a-3655-11e9-877b-0242ac110006', 'pipeline_result_path': '/ml_studio/model/a878b137-d9c7-4217-9b91-f09b7837d13d/b434073a-3655-11e9-877b-0242ac110006', 'metadata_path': '/ml_studio/metadata/a878b137-d9c7-4217-9b91-f09b7837d13d/b434073a-3655-11e9-877b-0242ac110006', 'spark_job_version': '1.1.2'}

from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression


def stage_result(data_train, message):
	"""
	Input: Dataset
	Output: Pipeline model.
	"""
	params_dict_LinearRegression = {'labelCol': 'price', 'featuresCol': '1_vector', 'predictionCol': 'prediction', 'aggregationDepth': 2, 'solver': 'auto', 'standardization': True, 'fitIntercept': True, 'elasticNetParam': 0, 'maxIter': 100, 'regParam': 0, 'tol': 1e-06, 'loss': 'squaredError', 'epsilon': 1.35}
	modul_LinearRegression = LinearRegression(**params_dict_LinearRegression)

	stages_pipeline = [modul_LinearRegression]

	pipeline = Pipeline(stages=stages_pipeline)
	model = pipeline.fit(data_train)

	prediction = model.transform(data_train)
	if message['output_col'] == 'all':
		data_prediction = prediction
	else:
		try:
			output_col = set(message['output_col'])
			prediction_col = set(prediction.columns)
			output_col = list(output_col.intersection(prediction_col))
			data_prediction = prediction.select([col for col in prediction.columns if col in output_col])
		except Exception as er:
			raise er

	return model, data_prediction


def get_result(message):
	#======================================================
	# General Pipeline
	#======================================================

	try:
		# Load and join data.
		logger.info("Starting load and join data.")
		data_train = get_and_join_data(message, spark)

		time_start = timeit.default_timer()
		n = 0

		while n < 3:
			try:
				logger.info("Starting process General Pipeline.")
				model, data_prediction = stage_result(data_train, message)
				break
			except Exception as err:
				n += 1
				logger.error("Error occured when processing General Pipeline with {0} trial.".format(n))
				if n == 3:
					raise err

		time_stop = timeit.default_timer()
		time_total = time_stop - time_start

		# Save model or data to hadoop
		from utils.hadoop.save_to_hadoop import save_model_to_hadoop
		hadoop_path_model = message["pipeline_result_path"]
		path = save_model_to_hadoop(model, hadoop_path_model)

	except Exception as er:
		raise er

	return model, data_prediction, data_train, path, time_total


def main(message):
	try:
		model, data_prediction, data_train, path, time_total = get_result(message)
		hadoop_metadata_path = message["metadata_path"]
		metadata = Metadata(spark, hadoop_metadata_path, PipelineType.FIT, None, data_prediction, data_train, model, message, time_total)
		metadata.process_metadata()

	except Exception as er:
		traceback.print_exc(file=sys.stdout)
		hadoop_metadata_path = message["metadata_path"]
		metadata = Metadata(spark, hadoop_metadata_path, PipelineType.FIT, str(er))
		metadata.process_metadata()
	spark.stop()


if __name__ == "__main__":
	main(pipeline)
