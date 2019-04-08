#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = """\n""".join(['Created by : ','Markus Paramahasti <markus@volantis.io>'])


from sparkflow.pipeline_util import PysparkPipelineWrapper
from pyspark.ml.pipeline import PipelineModel


# Load the model
p = PysparkPipelineWrapper.unwrap(PipelineModel.load('./your/model/location'))
# Use the model for inference
predictions = p.transform(df)
# Show the result of prediction
predictions.show()
