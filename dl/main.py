
from script_generator import generator
from spark_master import run_in_cluster
from pipeline_message_example import PIPELINE_MESSAGE

def runner(kafka_message: str):
    """
    This process is to generated script from incoming message.
    :param kafka_message: (str)
    :return:
    """

    # Convert message to dictionary
    #message_value = kafka_message.value()
    #message_str = message_value.decode("utf-8")
    #message_dict = json.loads(message_str)

    # Generate python script
    #filename = generator(message_dict)
    fn = generator(kafka_message)

    # Submit and running script in spark cluster
    #run_in_cluster(filename, message_dict)
    return filename

message = {
              "id": "b434073a-3655-11e9-877b-0242ac110006",
              "userId": "a878b137-d9c7-4217-9b91-f09b7837d13d",
              "jobId": None,
              "algorithmConfiguration": "{\"user_id\": \"a878b137-d9c7-4217-9b91-f09b7837d13d\", \"date\": \"2019-02-22T03:39:14.530Z\", \"username\": \"\", \"pipeline_group_name\": \"HPP with LR GP Testing 14042019\", \"composer_id\": \"79b09cd8-3653-11e9-be8a-0242ac110002\", \"parents_data_id\": [\"7c4670cf-0f49-4091-93b0-945ce8c67b54\"], \"parents_user_id_model\": [], \"parents_model_id\": [], \"parents_user_id_data\": [\"a878b137-d9c7-4217-9b91-f09b7837d13d\"], \"parents_data_price\": [0], \"parents_model_price\": [], \"numFolds\": 3, \"trainRatio\": 1, \"last_stage\": \"fit\", \"type\": \"GeneralPipeline\", \"output_col\": \"all\", \"estimatorParamMaps\": {\"ParamGridBuilder\": [{\"LinearRegression\": \"None\"}]}, \"evaluator\": {\"name\": \"MulticlassClassificationEvaluator\", \"params\": {\"metricName\": \"f1\"}}, \"estimator\": [{\"LinearRegression\": {\"labelCol\": \"price\", \"featuresCol\": \"1_vector\", \"predictionCol\": \"prediction\", \"aggregationDepth\": 2, \"solver\": \"auto\", \"standardization\": true, \"fitIntercept\": true, \"elasticNetParam\": 0, \"maxIter\": 100, \"regParam\": 0, \"tol\": 1e-06, \"loss\": \"squaredError\", \"epsilon\": 1.35}}], \"data\": {\"path\": [\"file:////Users/tmunir/Hobby/Volantis/Galaxy/hadoop/ml_studio/source_dataset/7c4670cf-0f49-4091-93b0-945ce8c67b54/1/1546403372058\"], \"action\": {\"adapter\": [[{\"from\": [{\"value\": \"n818134795\", \"datatype\": \"string\"}, {\"value\": \"n1624556946\", \"datatype\": \"string\"}, {\"value\": \"p1433103548\", \"datatype\": \"string\"}, {\"value\": \"n247383290\", \"datatype\": \"string\"}, {\"value\": \"n1552720724\", \"datatype\": \"string\"}], \"to\": [{\"value\": \"1_vector\", \"datatype\": \"vector\"}]}, {\"from\": [{\"value\": \"p77381929\", \"datatype\": \"string\"}], \"to\": [{\"value\": \"price\", \"datatype\": \"double\"}]}]], \"join\": \"None\"}, \"hash_map\": [[\"n818134795\", \"p77381929\", \"n1624556946\", \"p1433103548\", \"n247383290\", \"n1552720724\"]], \"hash_dict\": {\"n818134795\": {\"name\": \"size\", \"datatype\": \"double\"}, \"p77381929\": {\"name\": \"price\", \"datatype\": \"double\"}, \"n1624556946\": {\"name\": \"bathroom\", \"datatype\": \"int\"}, \"p1433103548\": {\"name\": \"bedroom\", \"datatype\": \"int\"}, \"n247383290\": {\"name\": \"garage_cars\", \"datatype\": \"int\"}, \"n1552720724\": {\"name\": \"land_size\", \"datatype\": \"int\"}}, \"input_col\": [\"1_vector\", \"price\"], \"column_input_dictionary\": {\"1_vector\": {\"datatype\": \"vector\", \"columns\": [\"size\", \"bathroom\", \"bedroom\", \"garage_cars\", \"land_size\"], \"columns_datatype\": [\"double\", \"int\", \"int\", \"int\", \"int\"]}, \"price\": {\"datatype\": \"double\"}}}, \"fit_id\": \"79b09ab2-3653-11e9-be8a-0242ac110002\", \"access_token\": \"EZ66yi6GZWbDOcvoooXajBWbCY3GrKcQwRRq83egVztRPDa6zg6z7is9zrbR3yoq\"}",
              "runningTime": None,
              "memoryConsumed": None,
              "outputPath": None,
              "inputCol": None,
              "outputCol": None,
              "params": None,
              "pipelineGroupName": "HPP with LR GP Testing 14042019",
              "metricPerformance": None,
              "parentsDataId": ["55a36855-be4e-11e8-92b8-08d40cec20c9"],
              "parentsDataUserId": ["b5f232f3-18b9-4a5a-b728-b2fcddfed955"],
              "parentsDataPrice": [0],
              "parentsModelId": None,
              "parentsModelUserId": None,
              "parentsModelPrice": None,
              "inputSchema": None,
              "outputSchema": None,
              "dataSampleModel": None,
              "payloadData": None,
              "size": None,
              "price": None,
              "endpoints": None,
              "endpointsServiceId": None,
              "logError": None,
              "status": {
                            "stage": "RUNNING",
                            "result": "QUEUED"
                },
                "pipelineType": "FIT",
        }

if __name__ == '__main__':

    runner(message)
    print (message)
