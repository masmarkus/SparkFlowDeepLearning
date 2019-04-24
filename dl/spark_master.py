def run_in_cluster(filename: str, pipeline_message: dict):
    """
    Submit script to spark cluster and monitored status job.
    :param filename: (str) script name
    :param pipeline_message: (dict) kafka payload message
    :return:
    """

    try:
        # Create hadoop connection
        hdfs = HDFileSystem(connect=False)
        hdfs.connect()

        # Submitting script
        submission_id = submit_script(filename, pipeline_message)

        # Update 'jobId'
        pipeline_message["jobId"] = submission_id

        # Monitoring status
        task(hdfs, submission_id, pipeline_message, filename)

        # Close hadoop connection
        hdfs.disconnect()

    except Exception as err:
        raise err
