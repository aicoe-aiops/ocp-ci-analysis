#!/usr/bin/env python
# coding: utf-8

# # Time to Merge Prediction Inference Service
# 
# In the previous notebook, we explored some basic machine learning models for predicting time to merge of a PR. We then deployed the model with the highest f1-score as a service using Seldon. The purpose of this notebook is to check whether this service is running as intended, and more specifically to ensure that the model performance is what we expect it to be. So here, we will use the test set from the aforementioned notebook as the query payload for the service, and then verify that the return values are the same as those obtained during training/testing locally.

# In[1]:


import os
import ast
import sys
import json
import datetime
from io import StringIO
import requests
from dotenv import load_dotenv, find_dotenv

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report

metric_template_path = "../../../notebooks/data-sources/TestGrid/metrics"
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
s3_path = "github/thoth"
REMOTE = os.getenv("REMOTE")
INPUT_DATA_PATH = "../../../data/processed/github"


# In[3]:


# read raw dataset
data_path = "../../data/raw/GitHub/thoth_PR_data.csv"

if REMOTE:
    print("getting dataset from ceph")
    cc = CephCommunication(s3_endpoint_url, s3_access_key, s3_secret_key, s3_bucket)
    s3_object = cc.s3_resource.Object(s3_bucket, "thoth_PR_data.csv")
    file = s3_object.get()["Body"].read().decode("utf-8")

pr_df = pd.read_csv(StringIO(file))


# In[4]:


# github pr dataset collected using thoth's mi-scheduler
pr_df.head()


# In[5]:


# remove PRs from train/test which are still open
pr_df = pr_df[pr_df["closed_at"].notna()]
pr_df = pr_df[pr_df["merged_at"].notna()]


# In[6]:


pr_df["created_at"] = pr_df["created_at"].apply(
    lambda x: int(datetime.datetime.timestamp(pd.to_datetime(x)))
)
pr_df["closed_at"] = pr_df["closed_at"].apply(
    lambda x: float(datetime.datetime.timestamp(pd.to_datetime(x)))
)
pr_df["merged_at"] = pr_df["merged_at"].apply(
    lambda x: float(datetime.datetime.timestamp(pd.to_datetime(x)))
)


# In[7]:


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


# In[8]:


X_test


# In[9]:


y_test


# In[10]:


# endpoint from the seldon deployment
base_url = "http://thoth-github-ttm-ds-ml-workflows-ws.apps.smaug.na.operate-first.cloud/predict"


# In[11]:


# lets extract the raw PR data corresponding to the PRs used in the test set
sample_payload = pr_df.reindex(X_test.index)


# In[12]:


sample_payload.head(2)


# In[13]:


sample_payload.changed_files = sample_payload.changed_files.apply(ast.literal_eval)


# In[14]:


sample_payload.dtypes


# In[15]:


sample_payload


# In[16]:


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


# In[17]:


class_dict = {
    0: "0 to 1 min",
    1: "1 to 2 mins",
    2: "2 to 8 mins",
    3: "8 to 20 mins",
    4: "20 mins to 1 hr",
    5: "1 to 4 hrs",
    6: "4 to 18 hrs",
    7: "18 hrs to 3 days",
    8: "3 days to 3 weeks",
    9: "more than 3 hrs",
}


# In[18]:


# query our inference service
response = requests.post(base_url, data=json_data, headers=headers)
response


# In[19]:


# what are the names of the prediction classes
json_response = response.json()
json_response["data"]["names"]


# In[20]:


sample_pr = 20


# In[21]:


# probabality estimates for each of the class for a sample PR
json_response["data"]["ndarray"][sample_pr][:10]


# In[22]:


# get predicted classes from probabilities for each PR
preds = np.argmax(json_response["data"]["ndarray"], axis=1)
print(
    "The PR belongs to class",
    preds[sample_pr],
    "and it is most likely to be merged in",
    class_dict[preds[sample_pr]],
)


# In[23]:


print("The PR was actually merged in", class_dict[int(y_test.iloc[sample_pr])])


# In[24]:


# evaluate results on the entire dataset
print(classification_report(y_test, preds))


# # Conclusion
# 
# This notebook shows how raw PR data can be sent to the deployed Seldon service to get time-to-merge predictions. Additionally, we see that the evaluation scores in the classification report match the ones we saw in the training notebook. So, great, looks like our inference service and model are working as expected, and are ready to predict some times to merge for GitHub PRs! 
