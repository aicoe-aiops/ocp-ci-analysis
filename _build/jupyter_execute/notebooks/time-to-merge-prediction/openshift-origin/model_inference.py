#!/usr/bin/env python
# coding: utf-8

# # Time to Merge Prediction Inference Service
# 
# In the previous notebook, we explored some basic machine learning models for predicting time to merge of a PR. We then deployed the model with the highest f1-score as a service using Seldon. The purpose of this notebook is to check whether this service is running as intended, and more specifically to ensure that the model performance is what we expect it to be. So here, we will use the test set from the aforementioned notebook as the query payload for the service, and then verify that the return values are the same as those obtained during training/testing locally.

# In[1]:


import os
import sys
import gzip
import json
import boto3
import requests
from dotenv import load_dotenv, find_dotenv

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report

metric_template_path = "../../data-sources/TestGrid/metrics"
if metric_template_path not in sys.path:
    sys.path.insert(1, metric_template_path)

from ipynb.fs.defs.metric_template import (  # noqa: E402
    CephCommunication,
)

load_dotenv(find_dotenv())


# In[2]:


## CEPH Bucket variables
## Create a .env file on your local with the correct configs,
s3_endpoint_url = os.getenv("S3_ENDPOINT")
s3_access_key = os.getenv("S3_ACCESS_KEY")
s3_secret_key = os.getenv("S3_SECRET_KEY")
s3_bucket = os.getenv("S3_BUCKET")
s3_path = "github"
REMOTE = os.getenv("REMOTE")
INPUT_DATA_PATH = "../../../data/processed/github"


# In[3]:


# read raw dataset
data_path = "../../data/raw/GitHub/PullRequest.json.gz"
OUTPUT_DATA_PATH = "../../data/processed/github"

if REMOTE:
    print("getting dataset from ceph")
    s3 = boto3.resource(
        "s3",
        endpoint_url=s3_endpoint_url,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
    )
    content = s3.Object(s3_bucket, "thoth/mi/openshift/origin/PullRequest.json")
    file = content.get()["Body"].read().decode("utf-8")
    prs = json.loads(file)

    with gzip.open(data_path, "wb") as out_file:
        out_file.write(json.dumps(prs).encode("utf-8"))

else:
    print("getting dataset from local")
    with gzip.open(data_path, "r") as f:
        prs = json.loads(f.read().decode("utf-8"))

pr_df = pd.DataFrame(prs).T


# In[4]:


# github pr dataset collected using thoth's mi-scheduler
pr_df.head()


# In[5]:


# read processed and split data created for train/test in the model training notebook
if REMOTE:
    cc = CephCommunication(s3_endpoint_url, s3_access_key, s3_secret_key, s3_bucket)
    X_test = cc.read_from_ceph(s3_path, "X_test.parquet")
    y_test = cc.read_from_ceph(s3_path, "y_test.parquet")

else:
    print(
        "The X_test.parquet and y_test.parquet files are not included in the ocp-ci-analysis github repo."
    )
    print(
        "Please set REMOTE=1 in the .env file and read this data from the S3 bucket instead."
    )


# In[6]:


# endpoint from the seldon deployment
base_url = (
    "http://github-pr-ttm-ds-ml-workflows-ws.apps.smaug.na.operate-first.cloud/predict"
)


# In[7]:


X_test


# In[8]:


# lets extract the raw PR data corresponding to the PRs used in the test set
sample_payload = pr_df.reindex(X_test.index)


# In[9]:


# convert the dataframe into a numpy array and then to a list (required by seldon)
data = {
    "data": {
        "names": sample_payload.columns.tolist(),
        "ndarray": sample_payload.to_numpy().tolist(),
    }
}

# create the query payload
json_data = json.dumps(data)
headers = {"content-Type": "application/json"}


# In[10]:


# query our inference service
response = requests.post(base_url, data=json_data, headers=headers)
response


# In[11]:


# what are the names of the prediction classes
json_response = response.json()
json_response["data"]["names"]


# In[12]:


# probabality estimates for each of the class for a sample PR
json_response["data"]["ndarray"][0][:10]


# In[13]:


# get predicted classes from probabilities for each PR
preds = np.argmax(json_response["data"]["ndarray"], axis=1)
preds[:10]


# In[14]:


# evaluate results
print(classification_report(y_test, preds))


# # Conclusion
# 
# This notebook shows how raw PR data can be sent to the deployed Seldon service to get time-to-merge predictions. Additionally, we see that the evaluation scores in the classification report match the ones we saw in the training notebook. So, great, looks like our inference service and model are working as expected, and are ready to predict some times to merge for GitHub PRs! 
