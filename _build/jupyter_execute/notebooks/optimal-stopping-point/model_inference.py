#!/usr/bin/env python
# coding: utf-8

# # Optimal Stopping Point Inference Service
# 
# In the previous notebook, we explored various distribution models to find the best model for predicting the optimal stopping point for a given test. We then deployed the model as a service using Seldon. The purpose of this notebook is to check whether this service is running as intended, and more specifically to ensure that the model performance is what we expect it to be. So here, we will use the test set from the aforementioned notebook as the query payload for the service, and then verify that the return values are the same as those obtained during training/testing locally.

# In[1]:


import os
import sys
import json
import requests
from dotenv import load_dotenv, find_dotenv
import datetime

metric_template_path = "../data-sources/TestGrid/metrics"
if metric_template_path not in sys.path:
    sys.path.insert(1, metric_template_path)

load_dotenv(find_dotenv())


# In[2]:


## CEPH Bucket variables
## Create a .env file on your local with the correct configs,
s3_endpoint_url = os.getenv("S3_ENDPOINT")
s3_access_key = os.getenv("S3_ACCESS_KEY")
s3_secret_key = os.getenv("S3_SECRET_KEY")
s3_bucket = os.getenv("S3_BUCKET")
s3_path = "osp"
REMOTE = os.getenv("REMOTE")


# In[3]:


# endpoint from the seldon deployment
base_url = "http://optimal-stopping-point-ds-ml-workflows-ws.apps.smaug.na.operate-first.cloud/predict"


# In[4]:


# Send request by adding a testname and timestamp.
data = {
    "jsonData": {
        "test_name": "operator.Run multi-stage test e2e-aws-upgrade - "
        "e2e-aws-upgrade-openshift-e2e-test container test",
        "timestamp": datetime.datetime(2021, 8, 24).timestamp(),
    }
}

# create the query payload
json_data = json.dumps(data)
headers = {"content-Type": "application/json"}


# In[5]:


# query our inference service
response = requests.post(base_url, data=json_data, headers=headers)
response


# In[6]:


response.json()


# # Conclusion
# 
# This notebook shows how test name and timestamp can be sent to the deployed Seldon service to get optimal-stopping-point predictions. So, great, looks like our inference service and model are working as expected, and are ready to predict some stopping times for the failing tests! 
