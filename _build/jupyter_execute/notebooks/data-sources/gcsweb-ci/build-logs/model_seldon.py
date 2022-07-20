#!/usr/bin/env python
# coding: utf-8

# # Seldon deployment for build log clustering
# In this notebook, we deploy a seldon service for clustering build logs. First, we take the experiments in [build log clustering notebook](build_log_term_freq.ipynb) and train a Sklearn pipeline with all the components. Then, we save the model on s3 storage and deploy a seldon service that uses the saved model. Finally, we test the service for inference on an example request. 

# In[1]:


import os
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import boto3
import json
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


# # Load Dataset

# In[2]:


# Note: periodic jobs only (see FIXME in class Builds)
job_name = "periodic-ci-openshift-release-master-ci-4.8-e2e-gcp"

logs_path = "../../../../data/raw/gcs/build-logs/"  # local cache of build log files
metadata_path = "../../../../data/raw/gcs/build-metadata/"  # path to saved metadata
metadata_file_name = os.path.join(metadata_path, f"{job_name}_build-logs.csv")


def log_path_for(build_id):
    return os.path.join(logs_path, f"{build_id}.txt")


def prow_url_for(build_id):
    project = "origin-ci-test"
    # FIXME: this prefix is only for periodic jobs
    job_prefix = f"logs/{job_name}/"
    return f"https://prow.ci.openshift.org/view/gcs/{project}/{job_prefix}{build_id}"


def clean_df(df):
    """Polishes the metadata DataFrame"""
    build_errors = df[df["result"] == "error"].index
    df.drop(build_errors, inplace=True)  # Remove builds that erroed (prow error)
    df["duration"] = df["end"] - df["start"]  # From timestamps to job duration
    df["success"] = df["result"] == "SUCCESS"  # A boolean version of the result
    return df


print("Reading metadata from", metadata_file_name)
df = pd.read_csv(metadata_file_name, index_col=0)
df = clean_df(df)
df


# In[3]:


# Get a list of paths to the local copy of each build log
build_logs = []
for build_id in df.index:
    with open(log_path_for(build_id), "r") as f:
        build_logs.append(f.read())


# # Train SKlearn Pipeline

# In[4]:


token_pattern = r"\b[a-z][a-z0-9_/\.-]+\b"
vectorizer = TfidfVectorizer(
    min_df=0.03,
    token_pattern=token_pattern,
)

k = 3
kmeans = KMeans(n_clusters=k, random_state=123)

pipeline = Pipeline([("tfidf", vectorizer), ("kmeans", kmeans)])


# In[5]:


pipeline.fit(build_logs)


# # Save Pipeline

# In[6]:


joblib.dump(pipeline, "model.joblib")


# In[7]:


# Test set
test_set = [i for i in build_logs if len(i) < 5000][:25]


# In[8]:


# Sanity check to see if the saved model works locally
pipeline_loaded = joblib.load("model2.joblib")
pipeline_loaded
pipeline_loaded.predict(test_set)


# In[10]:


# Set credentials for your s3 storage
s3_endpoint_url = os.getenv("S3_ENDPOINT")
aws_access_key_id = os.getenv("S3_ACCESS_KEY")
aws_secret_access_key = os.getenv("S3_SECRET_KEY")
s3_bucket = os.getenv("S3_BUCKET")


# In[13]:


s3_resource = boto3.resource(
    "s3",
    endpoint_url=s3_endpoint_url,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)
bucket = s3_resource.Bucket(name=s3_bucket)


# In[14]:


# Upload your model
bucket.upload_file(
    "model.joblib", "ai4ci/build-log-clustering/tfidf-kmeans/model.joblib"
)

# Check if your model exists on s3
objects = [
    obj.key for obj in bucket.objects.filter(Prefix="") if "model.joblib" in obj.key
]
objects


# # Test seldon deployment service 
# We use the deployment [config](seldon_deployment_config.yaml) to deploy a seldon service.

# In[15]:


# Service url
base_url = "http://build-log-clustering-ds-ml-workflows-ws.apps.smaug.na.operate-first.cloud/predict"


# In[16]:


# convert the dataframe into a numpy array and then to a list (required by seldon)
data = {"data": {"ndarray": test_set}}

# create the query payload
json_data = json.dumps(data)
headers = {"content-Type": "application/json"}

# query our inference service
response = requests.post(base_url, data=json_data, headers=headers)
response


# In[17]:


response.json()


# # Conclusion
# In this notebook, we saw how to create and save an unsupervised model for clustering build logs. We successfully deployed and tested the model using s3 for storage and a seldon service on Openshift. 
